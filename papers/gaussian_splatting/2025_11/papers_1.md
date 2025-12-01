# gaussian splatting - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20348v2">Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 8 pages, 5 figures. Submitted to IEEE Intelligent Vehicles Symposium (IV) 2026 for possible publication. Revised version (v2) to correct author order
    </div>
    <details class="paper-abstract">
      3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23292v1">FACT-GS: Frequency-Aligned Complexity-Aware Texture Reparameterization for 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 11 pages, 6 figures, preprint
    </div>
    <details class="paper-abstract">
      Realistic scene appearance modeling has advanced rapidly with Gaussian Splatting, which enables real-time, high-quality rendering. Recent advances introduced per-primitive textures that incorporate spatial color variations within each Gaussian, improving their expressiveness. However, texture-based Gaussians parameterize appearance with a uniform per-Gaussian sampling grid, allocating equal sampling density regardless of local visual complexity. This leads to inefficient texture space utilization, where high-frequency regions are under-sampled and smooth regions waste capacity, causing blurred appearance and loss of fine structural detail. We introduce FACT-GS, a Frequency-Aligned Complexity-aware Texture Gaussian Splatting framework that allocates texture sampling density according to local visual frequency. Grounded in adaptive sampling theory, FACT-GS reformulates texture parameterization as a differentiable sampling-density allocation problem, replacing the uniform textures with a learnable frequency-aware allocation strategy implemented via a deformation field whose Jacobian modulates local sampling density. Built on 2D Gaussian Splatting, FACT-GS performs non-uniform sampling on fixed-resolution texture grids, preserving real-time performance while recovering sharper high-frequency details under the same parameter budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09464v2">Hybrid Rendering for Multimodal Autonomous Driving: Merging Neural and Physics-Based Simulation</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Neural reconstruction models for autonomous driving simulation have made significant strides in recent years, with dynamic models becoming increasingly prevalent. However, these models are typically limited to handling in-domain objects closely following their original trajectories. We introduce a hybrid approach that combines the strengths of neural reconstruction with physics-based rendering. This method enables the virtual placement of traditional mesh-based dynamic agents at arbitrary locations, adjustments to environmental conditions, and rendering from novel camera viewpoints. Our approach significantly enhances novel view synthesis quality -- especially for road surfaces and lane markings -- while maintaining interactive frame rates through our novel training method, NeRF2GS. This technique leverages the superior generalization capabilities of NeRF-based methods and the real-time rendering speed of 3D Gaussian Splatting (3DGS). We achieve this by training a customized NeRF model on the original images with depth regularization derived from a noisy LiDAR point cloud, then using it as a teacher model for 3DGS training. This process ensures accurate depth, surface normals, and camera appearance modeling as supervision. With our block-based training parallelization, the method can handle large-scale reconstructions (greater than or equal to 100,000 square meters) and predict segmentation masks, surface normals, and depth maps. During simulation, it supports a rasterization-based rendering backend with depth-based composition and multiple camera models for real-time camera simulation, as well as a ray-traced backend for precise LiDAR simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23044v1">Geometry-Consistent 4D Gaussian Splatting for Sparse-Input Dynamic View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has been considered as a novel way for view synthesis of dynamic scenes, which shows great potential in AIoT applications such as digital twins. However, recent dynamic Gaussian Splatting methods significantly degrade when only sparse input views are available, limiting their applicability in practice. The issue arises from the incoherent learning of 4D geometry as input views decrease. This paper presents GC-4DGS, a novel framework that infuses geometric consistency into 4D Gaussian Splatting (4DGS), offering real-time and high-quality dynamic scene rendering from sparse input views. While learning-based Multi-View Stereo (MVS) and monocular depth estimators (MDEs) provide geometry priors, directly integrating these with 4DGS yields suboptimal results due to the ill-posed nature of sparse-input 4D geometric optimization. To address these problems, we introduce a dynamic consistency checking strategy to reduce estimation uncertainties of MVS across spacetime. Furthermore, we propose a global-local depth regularization approach to distill spatiotemporal-consistent geometric information from monocular depths, thereby enhancing the coherent geometry and appearance learning within the 4D volume. Extensive experiments on the popular N3DV and Technicolor datasets validate the effectiveness of GC-4DGS in rendering quality without sacrificing efficiency. Notably, our method outperforms RF-DeRF, the latest dynamic radiance field tailored for sparse-input dynamic view synthesis, and the original 4DGS by 2.62dB and 1.58dB in PSNR, respectively, with seamless deployability on resource-constrained IoT edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.23030v1">DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated impressive results for novel view synthesis with real-time rendering capabilities. However, integrating 3DGS with SLAM systems faces a fundamental scalability limitation: methods are constrained by GPU memory capacity, restricting reconstruction to small-scale environments. We present DiskChunGS, a scalable 3DGS SLAM system that overcomes this bottleneck through an out-of-core approach that partitions scenes into spatial chunks and maintains only active regions in GPU memory while storing inactive areas on disk. Our architecture integrates seamlessly with existing SLAM frameworks for pose estimation and loop closure, enabling globally consistent reconstruction at scale. We validate DiskChunGS on indoor scenes (Replica, TUM-RGBD), urban driving scenarios (KITTI), and resource-constrained Nvidia Jetson platforms. Our method uniquely completes all 11 KITTI sequences without memory failures while achieving superior visual quality, demonstrating that algorithmic innovation can overcome the memory constraints that have limited previous 3DGS SLAM methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.20789v2">LoDAvatar: Hierarchical Embedding and Selective Detail Enhancement for Adaptive Levels of Detail Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 21 pages, 7 figures, Published in Virtual Reality
    </div>
    <details class="paper-abstract">
      With the advancement of virtual reality, the demand for 3D human avatars is increasing. The emergence of Gaussian Splatting technology has enabled the rendering of Gaussian avatars with superior visual quality and reduced computational costs. Despite numerous methods researchers propose for implementing drivable Gaussian avatars, limited attention has been given to balancing visual quality and computational costs. In this paper, we introduce LoDAvatar, a method that introduces levels of detail into Gaussian avatars through hierarchical embedding and selective detail enhancement methods. The key steps of LoDAvatar encompass data preparation, Gaussian embedding, Gaussian optimization, and selective detail enhancement. We conducted experiments involving Gaussian avatars at various levels of detail, employing both objective assessments and subjective evaluations. The outcomes indicate that incorporating levels of detail into Gaussian avatars can decrease computational costs during rendering while upholding commendable visual quality, thereby enhancing runtime frame rates. We advocate adopting LoDAvatar to render multiple dynamic Gaussian avatars or extensive Gaussian scenes to balance visual quality and computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22997v1">MrGS: Multi-modal Radiance Fields with 3D Gaussian Splatting for RGB-Thermal Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-28
      | 💬 Accepted at Thermal Infrared in Robotics (TIRO) Workshop, ICRA 2025 (Best Poster Award)
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) have achieved considerable performance in RGB scene reconstruction. However, multi-modal rendering that incorporates thermal infrared imagery remains largely underexplored. Existing approaches tend to neglect distinctive thermal characteristics, such as heat conduction and the Lambertian property. In this study, we introduce MrGS, a multi-modal radiance field based on 3DGS that simultaneously reconstructs both RGB and thermal 3D scenes. Specifically, MrGS derives RGB- and thermal-related information from a single appearance feature through orthogonal feature extraction and employs view-dependent or view-independent embedding strategies depending on the degree of Lambertian reflectance exhibited by each modality. Furthermore, we leverage two physics-based principles to effectively model thermal-domain phenomena. First, we integrate Fourier's law of heat conduction prior to alpha blending to model intensity interpolation caused by thermal conduction between neighboring Gaussians. Second, we apply the Stefan-Boltzmann law and the inverse-square law to formulate a depth-aware thermal radiation map that imposes additional geometric constraints on thermal rendering. Experimental results demonstrate that the proposed MrGS achieves high-fidelity RGB-T scene reconstruction while reducing the number of Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22939v1">DenoiseGS: Gaussian Reconstruction Model for Burst Denoising</a></div>
    <div class="paper-meta">
      📅 2025-11-28
    </div>
    <details class="paper-abstract">
      Burst denoising methods are crucial for enhancing images captured on handheld devices, but they often struggle with large motion or suffer from prohibitive computational costs. In this paper, we propose DenoiseGS, the first framework to leverage the efficiency of 3D Gaussian Splatting for burst denoising. Our approach addresses two key challenges when applying feedforward Gaussian reconsturction model to noisy inputs: the degradation of Gaussian point clouds and the loss of fine details. To this end, we propose a Gaussian self-consistency (GSC) loss, which regularizes the geometry predicted from noisy inputs with high-quality Gaussian point clouds. These point clouds are generated from clean inputs by the same model that we are training, thereby alleviating potential bias or domain gaps. Additionally, we introduce a log-weighted frequency (LWF) loss to strengthen supervision within the spectral domain, effectively preserving fine-grained details. The LWF loss adaptively weights frequency discrepancies in a logarithmic manner, emphasizing challenging high-frequency details. Extensive experiments demonstrate that DenoiseGS significantly exceeds the state-of-the-art NeRF-based methods on both burst denoising and novel view synthesis under noisy conditions, while achieving \textbf{250$\times$} faster inference speed. Code and models are released at https://github.com/yscheng04/DenoiseGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22793v1">GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Channel state information (CSI) is essential for adaptive beamforming and maintaining robust links in wireless communication systems. However, acquiring CSI incurs significant overhead, consuming up to 25\% of spectrum resources in 5G networks due to frequent pilot transmissions at sub-millisecond intervals. Recent approaches aim to reduce this burden by reconstructing CSI from spatiotemporal RF measurements, such as signal strength and direction-of-arrival. While effective in offline settings, these methods often suffer from inference latencies in the 5--100~ms range, making them impractical for real-time systems. We present GSpaRC: Gaussian Splatting for Real-time Reconstruction of RF Channels, the first algorithm to break the 1 ms latency barrier while maintaining high accuracy. GSpaRC represents the RF environment using a compact set of 3D Gaussian primitives, each parameterized by a lightweight neural model augmented with physics-informed features such as distance-based attenuation. Unlike traditional vision-based splatting pipelines, GSpaRC is tailored for RF reception: it employs an equirectangular projection onto a hemispherical surface centered at the receiver to reflect omnidirectional antenna behavior. A custom CUDA pipeline enables fully parallelized directional sorting, splatting, and rendering across frequency and spatial dimensions. Evaluated on multiple RF datasets, GSpaRC achieves similar CSI reconstruction fidelity to recent state-of-the-art methods while reducing training and inference time by over an order of magnitude. By trading modest GPU computation for a substantial reduction in pilot overhead, GSpaRC enables scalable, low-latency channel estimation suitable for deployment in 5G and future wireless systems. The code is available here: \href{https://github.com/Nbhavyasai/GSpaRC-WirelessGaussianSplatting.git}{GSpaRC}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22704v1">Splat-SAP: Feed-Forward Gaussian Splatting for Human-Centered Scene with Scale-Aware Point Map Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026. Project page: https://yaourtb.github.io/Splat-SAP
    </div>
    <details class="paper-abstract">
      We present Splat-SAP, a feed-forward approach to render novel views of human-centered scenes from binocular cameras with large sparsity. Gaussian Splatting has shown its promising potential in rendering tasks, but it typically necessitates per-scene optimization with dense input views. Although some recent approaches achieve feed-forward Gaussian Splatting rendering through geometry priors obtained by multi-view stereo, such approaches still require largely overlapped input views to establish the geometry prior. To bridge this gap, we leverage pixel-wise point map reconstruction to represent geometry which is robust to large sparsity for its independent view modeling. In general, we propose a two-stage learning strategy. In stage 1, we transform the point map into real space via an iterative affinity learning process, which facilitates camera control in the following. In stage 2, we project point maps of two input views onto the target view plane and refine such geometry via stereo matching. Furthermore, we anchor Gaussian primitives on this refined plane in order to render high-quality images. As a metric representation, the scale-aware point map in stage 1 is trained in a self-supervised manner without 3D supervision and stage 2 is supervised with photo-metric loss. We collect multi-view human-centered data and demonstrate that our method improves both the stability of point map reconstruction and the visual quality of free-viewpoint rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.22070v3">FreeGaussian: Annotation-free Control of Articulated Objects via 3D Gaussian Splats with Flow Derivatives</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Reconstructing controllable Gaussian splats for articulated objects from monocular video is especially challenging due to its inherently insufficient constraints. Existing methods address this by relying on dense masks and manually defined control signals, limiting their real-world applications. In this paper, we propose an annotation-free method, FreeGaussian, which mathematically disentangles camera egomotion and articulated movements via flow derivatives. By establishing a connection between 2D flows and 3D Gaussian dynamic flow, our method enables optimization and continuity of dynamic Gaussian motions from flow priors without any control signals. Furthermore, we introduce a 3D spherical vector controlling scheme, which represents the state as a 3D Gaussian trajectory, thereby eliminating the need for complex 1D control signal calculations and simplifying controllable Gaussian modeling. Extensive experiments on articulated objects demonstrate the state-of-the-art visual performance and precise, part-aware controllability of our method. Code is available at: https://github.com/Tavish9/freegaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05296v2">Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Project webpage: https://galfiebelman.github.io/let-it-snow/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently enabled fast and photorealistic reconstruction of static 3D scenes. However, dynamic editing of such scenes remains a significant challenge. We introduce a novel framework, Physics-Guided Score Distillation, to address a fundamental conflict: physics simulation provides a strong motion prior that is insufficient for photorealism , while video-based Score Distillation Sampling (SDS) alone cannot generate coherent motion for complex, multi-particle scenarios. We resolve this through a unified optimization framework where physics simulation guides Score Distillation to jointly refine the motion prior for photorealism while simultaneously optimizing appearance. Specifically, we learn a neural dynamics model that predicts particle motion and appearance, optimized end-to-end via a combined loss integrating Video-SDS for photorealism with our physics-guidance prior. This allows for photorealistic refinements while ensuring the dynamics remain plausible. Our framework enables scene-wide dynamic weather effects, including snowfall, rainfall, fog, and sandstorms, with physically plausible motion. Experiments demonstrate our physics-guided approach significantly outperforms baselines, with ablations confirming this joint refinement is essential for generating coherent, high-fidelity dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22262v1">Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for 3D scenes, widely adopted due to its exceptional efficiency and high-fidelity visual quality. Given the significant value of 3DGS assets, recent works have introduced specialized watermarking schemes to ensure copyright protection and ownership verification. However, can existing 3D Gaussian watermarking approaches genuinely guarantee robust protection of the 3D assets? In this paper, for the first time, we systematically explore and validate possible vulnerabilities of 3DGS watermarking frameworks. We demonstrate that conventional watermark removal techniques designed for 2D images do not effectively generalize to the 3DGS scenario due to the specialized rendering pipeline and unique attributes of each gaussian primitives. Motivated by this insight, we propose GSPure, the first watermark purification framework specifically for 3DGS watermarking representations. By analyzing view-dependent rendering contributions and exploiting geometrically accurate feature clustering, GSPure precisely isolates and effectively removes watermark-related Gaussian primitives while preserving scene integrity. Extensive experiments demonstrate that our GSPure achieves the best watermark purification performance, reducing watermark PSNR by up to 16.34dB while minimizing degradation to original scene fidelity with less than 1dB PSNR loss. Moreover, it consistently outperforms existing methods in both effectiveness and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22233v1">IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Reconstructing high-resolution (HR) 3D Gaussian Splatting (3DGS) models from low-resolution (LR) inputs remains challenging due to the lack of fine-grained textures and geometry. Existing methods typically rely on pre-trained 2D super-resolution (2DSR) models to enhance textures, but suffer from 3D Gaussian ambiguity arising from cross-view inconsistencies and domain gaps inherent in 2DSR models. We propose IE-SRGS, a novel 3DGS SR paradigm that addresses this issue by jointly leveraging the complementary strengths of external 2DSR priors and internal 3DGS features. Specifically, we use 2DSR and depth estimation models to generate HR images and depth maps as external knowledge, and employ multi-scale 3DGS models to produce cross-view consistent, domain-adaptive counterparts as internal knowledge. A mask-guided fusion strategy is introduced to integrate these two sources and synergistically exploit their complementary strengths, effectively guiding the 3D Gaussian optimization toward high-fidelity reconstruction. Extensive experiments on both synthetic and real-world benchmarks show that IE-SRGS consistently outperforms state-of-the-art methods in both quantitative accuracy and visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22228v1">3D-Consistent Multi-View Editing by Diffusion Guidance</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      Recent advancements in diffusion models have greatly improved text-based image editing, yet methods that edit images independently often produce geometrically and photometrically inconsistent results across different views of the same scene. Such inconsistencies are particularly problematic for editing of 3D representations such as NeRFs or Gaussian Splat models. We propose a training-free diffusion framework that enforces multi-view consistency during the image editing process. The key assumption is that corresponding points in the unedited images should undergo similar transformations after editing. To achieve this, we introduce a consistency loss that guides the diffusion sampling toward coherent edits. The framework is flexible and can be combined with widely varying image editing methods, supporting both dense and sparse multi-view editing setups. Experimental results show that our approach significantly improves 3D consistency compared to existing multi-view editing methods. We also show that this increased consistency enables high-quality Gaussian Splat editing with sharp details and strong fidelity to user-specified text prompts. Please refer to our project page for video results: https://3d-consistent-editing.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22147v1">RemedyGS: Defend 3D Gaussian Splatting against Computation Cost Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      As a mainstream technique for 3D reconstruction, 3D Gaussian splatting (3DGS) has been applied in a wide range of applications and services. Recent studies have revealed critical vulnerabilities in this pipeline and introduced computation cost attacks that lead to malicious resource occupancies and even denial-of-service (DoS) conditions, thereby hindering the reliable deployment of 3DGS. In this paper, we propose the first effective and comprehensive black-box defense framework, named RemedyGS, against such computation cost attacks, safeguarding 3DGS reconstruction systems and services. Our pipeline comprises two key components: a detector to identify the attacked input images with poisoned textures and a purifier to recover the benign images from their attacked counterparts, mitigating the adverse effects of these attacks. Moreover, we incorporate adversarial training into the purifier to enforce distributional alignment between the recovered and original natural images, thereby enhancing the defense efficacy. Experimental results demonstrate that our framework effectively defends against white-box, black-box, and adaptive attacks in 3DGS systems, achieving state-of-the-art performance in both safety and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22056v1">EAST: Environment-Aware Stylized Transition Along the Reality-Virtuality Continuum</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      In the Virtual Reality (VR) gaming industry, maintaining immersion during real-world interruptions remains a challenge, particularly during transitions along the reality-virtuality continuum (RVC). Existing methods tend to rely on digital replicas or simple visual transitions, neglecting to address the aesthetic discontinuities between real and virtual environments, especially in highly stylized VR games. This paper introduces the Environment-Aware Stylized Transition (EAST) framework, which employs a novel style-transferred 3D Gaussian Splatting (3DGS) technique to transfer real-world interruptions into the virtual environment with seamless aesthetic consistency. Rather than merely transforming the real world into game-like visuals, EAST minimizes the disruptive impact of interruptions by integrating real-world elements within the framework. Qualitative user studies demonstrate significant enhancements in cognitive comfort and emotional continuity during transitions, while quantitative experiments highlight EAST's ability to maintain visual coherence across diverse VR styles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18755v2">Splatonic: Architecture Support for 3D Gaussian Splatting SLAM via Sparse Processing</a></div>
    <div class="paper-meta">
      📅 2025-11-27
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has emerged as a promising direction for SLAM due to its high-fidelity reconstruction and rapid convergence. However, 3DGS-SLAM algorithms remain impractical for mobile platforms due to their high computational cost, especially for their tracking process. This work introduces Splatonic, a sparse and efficient real-time 3DGS-SLAM algorithm-hardware co-design for resource-constrained devices. Inspired by classical SLAMs, we propose an adaptive sparse pixel sampling algorithm that reduces the number of rendered pixels by up to 256$\times$ while retaining accuracy. To unlock this performance potential on mobile GPUs, we design a novel pixel-based rendering pipeline that improves hardware utilization via Gaussian-parallel rendering and preemptive $α$-checking. Together, these optimizations yield up to 121.7$\times$ speedup on the bottleneck stages and 14.6$\times$ end-to-end speedup on off-the-shelf GPUs. To further address new bottlenecks introduced by our rendering pipeline, we propose a pipelined architecture that simplifies the overall design while addressing newly emerged bottlenecks in projection and aggregation. Evaluated across four 3DGS-SLAM algorithms, Splatonic achieves up to 274.9$\times$ speedup and 4738.5$\times$ energy savings over mobile GPUs and up to 25.2$\times$ speedup and 241.1$\times$ energy savings over state-of-the-art accelerators, all with comparable accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19854v2">STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-27
      | 💬 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21459v1">Resolution Where It Counts: Hash-based GPU-Accelerated 3D Reconstruction via Variance-Adaptive Voxel Grids</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted for publication in ACM Transaction on Graphics. Project site: https://rvp-group.github.io/mrhash/
    </div>
    <details class="paper-abstract">
      Efficient and scalable 3D surface reconstruction from range data remains a core challenge in computer graphics and vision, particularly in real-time and resource-constrained scenarios. Traditional volumetric methods based on fixed-resolution voxel grids or hierarchical structures like octrees often suffer from memory inefficiency, computational overhead, and a lack of GPU support. We propose a novel variance-adaptive, multi-resolution voxel grid that dynamically adjusts voxel size based on the local variance of signed distance field (SDF) observations. Unlike prior multi-resolution approaches that rely on recursive octree structures, our method leverages a flat spatial hash table to store all voxel blocks, supporting constant-time access and full GPU parallelism. This design enables high memory efficiency and real-time scalability. We further demonstrate how our representation supports GPU-accelerated rendering through a parallel quad-tree structure for Gaussian Splatting, enabling effective control over splat density. Our open-source CUDA/C++ implementation achieves up to 13x speedup and 4x lower memory usage compared to fixed-resolution baselines, while maintaining on par results in terms of reconstruction accuracy, offering a practical and extensible solution for high-performance 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11473v2">VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently emerged as an efficient solution for high-quality and real-time novel view synthesis. However, its capability for accurate surface reconstruction remains underexplored. Due to the discrete and unstructured nature of Gaussians, supervision based solely on image rendering loss often leads to inaccurate geometry and inconsistent multi-view alignment. In this work, we propose a novel method that enhances the geometric representation of 3D Gaussians through view alignment (VA). Specifically, we incorporate edge-aware image cues into the rendering loss to improve surface boundary delineation. To enforce geometric consistency across views, we introduce a visibility-aware photometric alignment loss that models occlusions and encourages accurate spatial relationships among Gaussians. To further mitigate ambiguities caused by lighting variations, we incorporate normal-based constraints to refine the spatial orientation of Gaussians and improve local surface estimation. Additionally, we leverage deep image feature embeddings to enforce cross-view consistency, enhancing the robustness of the learned geometry under varying viewpoints and illumination. Extensive experiments on standard benchmarks demonstrate that our method achieves state-of-the-art performance in both surface reconstruction and novel view synthesis. The source code is available at https://github.com/LeoQLi/VA-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21367v1">Endo-G$^{2}$T: Geometry-Guided & Temporally Aware Time-Embedded 4DGS For Endoscopic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Endoscopic (endo) video exhibits strong view-dependent effects such as specularities, wet reflections, and occlusions. Pure photometric supervision misaligns with geometry and triggers early geometric drift, where erroneous shapes are reinforced during densification and become hard to correct. We ask how to anchor geometry early for 4D Gaussian splatting (4DGS) while maintaining temporal consistency and efficiency in dynamic endoscopic scenes. Thus, we present Endo-G$^{2}$T, a geometry-guided and temporally aware training scheme for time-embedded 4DGS. First, geo-guided prior distillation converts confidence-gated monocular depth into supervision with scale-invariant depth and depth-gradient losses, using a warm-up-to-cap schedule to inject priors softly and avoid early overfitting. Second, a time-embedded Gaussian field represents dynamics in XYZT with a rotor-like rotation parameterization, yielding temporally coherent geometry with lightweight regularization that favors smooth motion and crisp opacity boundaries. Third, keyframe-constrained streaming improves efficiency and long-horizon stability through keyframe-focused optimization under a max-points budget, while non-keyframes advance with lightweight updates. Across EndoNeRF and StereoMIS-P1 datasets, Endo-G$^{2}$T achieves state-of-the-art results among monocular reconstruction baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21265v1">Unlocking Zero-shot Potential of Semi-dense Image Matching via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-26
    </div>
    <details class="paper-abstract">
      Learning-based image matching critically depends on large-scale, diverse, and geometrically accurate training data. 3D Gaussian Splatting (3DGS) enables photorealistic novel-view synthesis and thus is attractive for data generation. However, its geometric inaccuracies and biased depth rendering currently prevent robust correspondence labeling. To address this, we introduce MatchGS, the first framework designed to systematically correct and leverage 3DGS for robust, zero-shot image matching. Our approach is twofold: (1) a geometrically-faithful data generation pipeline that refines 3DGS geometry to produce highly precise correspondence labels, enabling the synthesis of a vast and diverse range of viewpoints without compromising rendering fidelity; and (2) a 2D-3D representation alignment strategy that infuses 3DGS' explicit 3D knowledge into the 2D matcher, guiding 2D semi-dense matchers to learn viewpoint-invariant 3D representations. Our generated ground-truth correspondences reduce the epipolar error by up to 40 times compared to existing datasets, enable supervision under extreme viewpoint changes, and provide self-supervisory signals through Gaussian attributes. Consequently, state-of-the-art matchers trained solely on our data achieve significant zero-shot performance gains on public benchmarks, with improvements of up to 17.7%. Our work demonstrates that with proper geometric refinement, 3DGS can serve as a scalable, high-fidelity, and structurally-rich data source, paving the way for a new generation of robust zero-shot image matchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07856v2">XYZCylinder: Towards Compatible Feed-Forward 3D Gaussian Splatting for Driving Scenes via Unified Cylinder Lifting Method</a></div>
    <div class="paper-meta">
      📅 2025-11-26
      | 💬 Feed-Forward, 3D Gaussian Splatting, Project page: https://yuyuyu223.github.io/XYZCYlinder-projectpage/
    </div>
    <details class="paper-abstract">
      Feed-forward paradigms for 3D reconstruction have become a focus of recent research, which learn implicit, fixed view transformations to generate a single scene representation. However, their application to complex driving scenes reveals significant limitations. Two core challenges are responsible for this performance gap. First, the reliance on a fixed view transformation hinders compatibility to varying camera configurations. Second, the inherent difficulty of learning complex driving scenes from sparse 360° views with minimal overlap compromises the final reconstruction fidelity. To handle these difficulties, we introduce XYZCylinder, a novel method built upon a unified cylinder lifting method that integrates camera modeling and feature lifting. To tackle the compatibility problem, we design a Unified Cylinder Camera Modeling (UCCM) strategy. This strategy explicitly models projection parameters to unify diverse camera setups, thus bypassing the need for learning viewpoint-dependent correspondences. To improve the reconstruction accuracy, we propose a hybrid representation with several dedicated modules based on newly designed Cylinder Plane Feature Group (CPFG) to lift 2D image features to 3D space. Extensive evaluations confirm that XYZCylinder not only achieves state-of-the-art performance under different evaluation settings but also demonstrates remarkable compatibility in entirely new scenes with different camera settings in a zero-shot manner. Project page: \href{https://yuyuyu223.github.io/XYZCYlinder-projectpage/}{here}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04283v2">FastGS: Training 3D Gaussian Splatting in 100 Seconds</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Project page: https://fastgs.github.io/
    </div>
    <details class="paper-abstract">
      The dominant 3D Gaussian splatting (3DGS) acceleration methods fail to properly regulate the number of Gaussians during training, causing redundant computational time overhead. In this paper, we propose FastGS, a novel, simple, and general acceleration framework that fully considers the importance of each Gaussian based on multi-view consistency, efficiently solving the trade-off between training time and rendering quality. We innovatively design a densification and pruning strategy based on multi-view consistency, dispensing with the budgeting mechanism. Extensive experiments on Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets demonstrate that our method significantly outperforms the state-of-the-art methods in training speed, achieving a 3.32$\times$ training acceleration and comparable rendering quality compared with DashGaussian on the Mip-NeRF 360 dataset and a 15.45$\times$ acceleration compared with vanilla 3DGS on the Deep Blending dataset. We demonstrate that FastGS exhibits strong generality, delivering 2-7$\times$ training acceleration across various tasks, including dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and simultaneous localization and mapping. The project page is available at https://fastgs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20354v1">GS-Checker: Tampering Localization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      Recent advances in editing technologies for 3D Gaussian Splatting (3DGS) have made it simple to manipulate 3D scenes. However, these technologies raise concerns about potential malicious manipulation of 3D content. To avoid such malicious applications, localizing tampered regions becomes crucial. In this paper, we propose GS-Checker, a novel method for locating tampered areas in 3DGS models. Our approach integrates a 3D tampering attribute into the 3D Gaussian parameters to indicate whether the Gaussian has been tampered. Additionally, we design a 3D contrastive mechanism by comparing the similarity of key attributes between 3D Gaussians to seek tampering cues at 3D level. Furthermore, we introduce a cyclic optimization strategy to refine the 3D tampering attribute, enabling more accurate tampering localization. Notably, our approach does not require expensive 3D labels for supervision. Extensive experimental results demonstrate the effectiveness of our proposed method to locate the tampered 3DGS area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20348v1">Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 8 pages, 5 figures. Submitted to IEEE Intelligent Vehicles Symposium (IV) 2026 for possible publication
    </div>
    <details class="paper-abstract">
      3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17811v2">MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Surface reconstruction has been widely studied in computer vision and graphics. However, existing surface reconstruction works struggle to recover accurate scene geometry when the input views are extremely sparse. To address this issue, we propose MeshSplat, a generalizable sparse-view surface reconstruction framework via Gaussian Splatting. Our key idea is to leverage 2DGS as a bridge, which connects novel view synthesis to learned geometric priors and then transfers these priors to achieve surface reconstruction. Specifically, we incorporate a feed-forward network to predict per-view pixel-aligned 2DGS, which enables the network to synthesize novel view images and thus eliminates the need for direct 3D ground-truth supervision. To improve the accuracy of 2DGS position and orientation prediction, we propose a Weighted Chamfer Distance Loss to regularize the depth maps, especially in overlapping areas of input views, and also a normal prediction network to align the orientation of 2DGS with normal vectors predicted by a monocular normal estimator. Extensive experiments validate the effectiveness of our proposed improvement, demonstrating that our method achieves state-of-the-art performance in generalizable sparse-view mesh reconstruction tasks. Project Page: https://hanzhichang.github.io/meshsplat_web
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19786v2">MAPo : Motion-Aware Partitioning of Deformable 3D Gaussian Splatting for High-Fidelity Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting, known for enabling high-quality static scene reconstruction with fast rendering, is increasingly being applied to multi-view dynamic scene reconstruction. A common strategy involves learning a deformation field to model the temporal changes of a canonical set of 3D Gaussians. However, these deformation-based methods often produce blurred renderings and lose fine motion details in highly dynamic regions due to the inherent limitations of a single, unified model in representing diverse motion patterns. To address these challenges, we introduce Motion-Aware Partitioning of Deformable 3D Gaussian Splatting (MAPo), a novel framework for high-fidelity dynamic scene reconstruction. Its core is a dynamic score-based partitioning strategy that distinguishes between high- and low-dynamic 3D Gaussians. For high-dynamic 3D Gaussians, we recursively partition them temporally and duplicate their deformation networks for each new temporal segment, enabling specialized modeling to capture intricate motion details. Concurrently, low-dynamic 3DGs are treated as static to reduce computational costs. However, this temporal partitioning strategy for high-dynamic 3DGs can introduce visual discontinuities across frames at the partition boundaries. To address this, we introduce a cross-frame consistency loss, which not only ensures visual continuity but also further enhances rendering quality. Extensive experiments demonstrate that MAPo achieves superior rendering quality compared to baselines while maintaining comparable computational costs, particularly in regions with complex or rapid motions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.05700v2">Temporally Compressed 3D Gaussian Splatting for Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted at British Machine Vision Conference (BMVC) 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in high-fidelity dynamic scene reconstruction have leveraged dynamic 3D Gaussians and 4D Gaussian Splatting for realistic scene representation. However, to make these methods viable for real-time applications such as AR/VR, gaming, and rendering on low-power devices, substantial reductions in memory usage and improvements in rendering efficiency are required. While many state-of-the-art methods prioritize lightweight implementations, they struggle in handling {scenes with complex motions or long sequences}. In this work, we introduce Temporally Compressed 3D Gaussian Splatting (TC3DGS), a novel technique designed specifically to effectively compress dynamic 3D Gaussian representations. TC3DGS selectively prunes Gaussians based on their temporal relevance and employs gradient-aware mixed-precision quantization to dynamically compress Gaussian parameters. In addition, TC3DGS exploits an adapted version of the Ramer-Douglas-Peucker algorithm to further reduce storage by interpolating Gaussian trajectories across frames. Our experiments on multiple datasets demonstrate that TC3DGS achieves up to 67$\times$ compression with minimal or no degradation in visual quality. More results and videos are provided in the supplementary. Project Page: https://ahmad-jarrar.github.io/tc-3dgs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17951v3">SplatCo: Structure-View Collaborative Gaussian Splatting for Detail-Preserving Rendering of Large-Scale Unbounded Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      We present SplatCo, a structure-view collaborative Gaussian splatting framework for high-fidelity rendering of complex outdoor environments. SplatCo builds upon two novel components: (1) a cross-structure collaboration module that combines global tri-plane representations, which capture coarse scene layouts, with local context grid features that represent fine surface details. This fusion is achieved through a novel hierarchical compensation strategy, ensuring both global consistency and local detail preservation; and (2) a cross-view assisted training strategy that enhances multi-view consistency by synchronizing gradient updates across viewpoints, applying visibility-aware densification, and pruning overfitted or inaccurate Gaussians based on structural consistency. Through joint optimization of structural representation and multi-view coherence, SplatCo effectively reconstructs fine-grained geometric structures and complex textures in large-scale scenes. Comprehensive evaluations on 13 diverse large-scale scenes, including Mill19, MatrixCity, Tanks & Temples, WHU, and custom aerial captures, demonstrate that SplatCo consistently achieves higher reconstruction quality than state-of-the-art methods, with PSNR improvements of 1-2 dB and SSIM gains of 0.1 to 0.2. These results establish a new benchmark for high-fidelity rendering of large-scale unbounded scenes. Code and additional information are available at https://github.com/SCUT-BIP-Lab/SplatCo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15447v3">LiHi-GS: LiDAR-Supervised Gaussian Splatting for Highway Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 RA-L 2025
    </div>
    <details class="paper-abstract">
      Photorealistic 3D scene reconstruction plays an important role in autonomous driving, enabling the generation of novel data from existing datasets to simulate safety-critical scenarios and expand training data without additional acquisition costs. Gaussian Splatting (GS) facilitates real-time, photorealistic rendering with an explicit 3D Gaussian representation of the scene, providing faster processing and more intuitive scene editing than the implicit Neural Radiance Fields (NeRFs). While extensive GS research has yielded promising advancements in autonomous driving applications, they overlook two critical aspects: First, existing methods mainly focus on low-speed and feature-rich urban scenes and ignore the fact that highway scenarios play a significant role in autonomous driving. Second, while LiDARs are commonplace in autonomous driving platforms, existing methods learn primarily from images and use LiDAR only for initial estimates or without precise sensor modeling, thus missing out on leveraging the rich depth information LiDAR offers and limiting the ability to synthesize LiDAR data. In this paper, we propose a novel GS method for dynamic scene synthesis and editing with improved scene reconstruction through LiDAR supervision and support for LiDAR rendering. Unlike prior works that are tested mostly on urban datasets, to the best of our knowledge, we are the first to focus on the more challenging and highly relevant highway scenes for autonomous driving, with sparse sensor views and monotone backgrounds. Visit our project page at: https://umautobots.github.io/lihi_gs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13186v3">STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control</a></div>
    <div class="paper-meta">
      📅 2025-11-25
    </div>
    <details class="paper-abstract">
      Edge Gaussian splatting (EGS), which aggregates data from distributed clients (e.g., drones) and trains a global GS model at the edge (e.g., ground server), is an emerging paradigm for scene reconstruction in low-altitude economy. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead.Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments reveal that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. The GS-oriented objective can be accurately predicted with low sampling ratios (e.g., 10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19861v1">GigaWorld-0: World Models as Data Engine to Empower Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Project Page: https://gigaworld0.github.io/
    </div>
    <details class="paper-abstract">
      World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple dimensions. Critically, VLA model (e.g., GigaBrain-0) trained on GigaWorld-0-generated data achieve strong real-world performance, significantly improving generalization and task success on physical robots without any real-world interaction during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19854v1">STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Reconstructing high-fidelity and animatable 3D head avatars from monocular videos remains a challenging yet essential task. Existing methods based on 3D Gaussian Splatting typically bind Gaussians to mesh triangles and model deformations solely via Linear Blend Skinning, which results in rigid motion and limited expressiveness. Moreover, they lack specialized strategies to handle frequently occluded regions (e.g., mouth interiors, eyelids). To address these limitations, we propose STAvatar, which consists of two key components: (1) a UV-Adaptive Soft Binding framework that leverages both image-based and geometric priors to learn per-Gaussian feature offsets within the UV space. This UV representation supports dynamic resampling, ensuring full compatibility with Adaptive Density Control (ADC) and enhanced adaptability to shape and textural variations. (2) a Temporal ADC strategy, which first clusters structurally similar frames to facilitate more targeted computation of the densification criterion. It further introduces a novel fused perceptual error as clone criterion to jointly capture geometric and textural discrepancies, encouraging densification in regions requiring finer details. Extensive experiments on four benchmark datasets demonstrate that STAvatar achieves state-of-the-art reconstruction performance, especially in capturing fine-grained details and reconstructing frequently occluded regions. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22908v2">Learning Hierarchical Sparse Transform Coding of 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Our code will be released at \href{https://github.com/hxu160/SHTC_for_3DGS_compression}{here}
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) supports fast, high quality, novel view synthesis but has a heavy memory footprint, making the compression of its model crucial. Current state-of-the-art (SOTA) 3DGS compression methods adopt an anchor-based architecture that pairs the Scaffold-GS representation with conditional entropy coding. However, these methods forego the analysis-synthesis transform, a vital mechanism in visual data compression. As a result, redundancy remains intact in the signal and its removal is left to the entropy coder, which computationally overburdens the entropy coding module, increasing coding latency. Even with added complexity thorough redundancy removal is a task unsuited to an entropy coder. To fix this critical omission, we introduce a Sparsity-guided Hierarchical Transform Coding (SHTC) method, the first study on the end-to-end learned neural transform coding of 3DGS. SHTC applies KLT to decorrelate intra-anchor attributes, followed by quantization and entropy coding, and then compresses KLT residuals with a low-complexity, scene-adaptive neural transform. Aided by the sparsity prior and deep unfolding technique, the learned transform uses only a few trainable parameters, reducing the memory usage. Overall, SHTC achieves an appreciably improved R-D performance and at the same time higher decoding speed over SOTA. Its prior-guided, parameter-efficient design may also inspire low-complexity neural image and video codecs. Our code will be released at https://github.com/hxu160/SHTC_for_3DGS_compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21890v3">Diffusion-Denoised Hyperspectral Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 Accepted to 3DV 2026
    </div>
    <details class="paper-abstract">
      Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise quantification of sample nutritional elements. Recently, 3D reconstruction methods, such as Neural Radiance Field (NeRF), have been used to create implicit neural representations of HSI scenes. This capability enables the rendering of hyperspectral channel compositions at every spatial location, thereby helping localize the target object's nutrient composition both spatially and spectrally. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of the hyperspectral scenes for the entire spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of our DD-HGS. The results demonstrate that DD-HGS achieves the new state-of-the-art performance compared to all the previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18389v2">FastAvatar: Instant 3D Gaussian Splatting for Faces from Single Unconstrained Poses</a></div>
    <div class="paper-meta">
      📅 2025-11-25
      | 💬 11 pages, 5 figures, website: https://hliang2.github.io/FastAvatar/
    </div>
    <details class="paper-abstract">
      We present FastAvatar, a fast and robust algorithm for single-image 3D face reconstruction using 3D Gaussian Splatting (3DGS). Given a single input image from an arbitrary pose, FastAvatar recovers a high-quality, full-head 3DGS avatar in approximately 3 seconds on a single NVIDIA A100 GPU. We use a two-stage design: a feed-forward encoder-decoder predicts coarse face geometry by regressing Gaussian structure from a pose-invariant identity embedding, and a lightweight test-time refinement stage then optimizes the appearance parameters for photorealistic rendering. This hybrid strategy combines the speed and stability of direct prediction with the accuracy of optimization, enabling strong identity preservation even under extreme input poses. FastAvatar achieves state-of-the-art reconstruction quality (24.01 dB PSNR, 0.91 SSIM) while running over 600x faster than existing per-subject optimization methods (e.g., FlashAvatar, GaussianAvatars, GASP). Once reconstructed, our avatars support photorealistic novel-view synthesis and FLAME-guided expression animation, enabling controllable reenactment from a single image. By jointly offering high fidelity, robustness to pose, and rapid reconstruction, FastAvatar significantly broadens the applicability of 3DGS-based facial avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19294v1">DensifyBeforehand: LiDAR-assisted Content-aware Densification for Efficient and Quality 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      This paper addresses the limitations of existing 3D Gaussian Splatting (3DGS) methods, particularly their reliance on adaptive density control, which can lead to floating artifacts and inefficient resource usage. We propose a novel densify beforehand approach that enhances the initialization of 3D scenes by combining sparse LiDAR data with monocular depth estimation from corresponding RGB images. Our ROI-aware sampling scheme prioritizes semantically and geometrically important regions, yielding a dense point cloud that improves visual fidelity and computational efficiency. This densify beforehand approach bypasses the adaptive density control that may introduce redundant Gaussians in the original pipeline, allowing the optimization to focus on the other attributes of 3D Gaussian primitives, reducing overlap while enhancing visual quality. Our method achieves comparable results to state-of-the-art techniques while significantly lowering resource consumption and training time. We validate our approach through extensive comparisons and ablation studies on four newly collected datasets, showcasing its effectiveness in preserving regions of interest in complex scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05813v2">Optimization-Free Style Transfer for 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      The task of style transfer for 3D Gaussian splats has been explored in many previous works, but these require reconstructing or fine-tuning the splat while incorporating style information or optimizing a feature extraction network on the splat representation. We propose a reconstruction- and optimization-free approach to stylizing 3D Gaussian splats, allowing for direct stylization on a .ply or .splat file without requiring the original camera views. This is done by generating a graph structure across the implicit surface of the splat representation. A feed-forward, surface-based stylization method is then used and interpolated back to the individual splats in the scene. This also allows for fast stylization of splats with no additional training, achieving speeds under 2 minutes even on CPU-based consumer hardware. We demonstrate the quality results this approach achieves and compare to other 3D Gaussian splat style transfer methods. Code is publicly available at https://github.com/davidmhart/FastSplatStyler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19235v1">IDSplat: Instance-Decomposed 3D Gaussian Splatting for Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic driving scenes is essential for developing autonomous systems through sensor-realistic simulation. Although recent methods achieve high-fidelity reconstructions, they either rely on costly human annotations for object trajectories or use time-varying representations without explicit object-level decomposition, leading to intertwined static and dynamic elements that hinder scene separation. We present IDSplat, a self-supervised 3D Gaussian Splatting framework that reconstructs dynamic scenes with explicit instance decomposition and learnable motion trajectories, without requiring human annotations. Our key insight is to model dynamic objects as coherent instances undergoing rigid transformations, rather than unstructured time-varying primitives. For instance decomposition, we employ zero-shot, language-grounded video tracking anchored to 3D using lidar, and estimate consistent poses via feature correspondences. We introduce a coordinated-turn smoothing scheme to obtain temporally and physically consistent motion trajectories, mitigating pose misalignments and tracking failures, followed by joint optimization of object poses and Gaussian parameters. Experiments on the Waymo Open Dataset demonstrate that our method achieves competitive reconstruction quality while maintaining instance-level decomposition and generalizes across diverse sequences and view densities without retraining, making it practical for large-scale autonomous driving applications. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19202v1">NVGS: Neural Visibility for Occlusion Culling in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 15 pages, 13 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting can exploit frustum culling and level-of-detail strategies to accelerate rendering of scenes containing a large number of primitives. However, the semi-transparent nature of Gaussians prevents the application of another highly effective technique: occlusion culling. We address this limitation by proposing a novel method to learn the viewpoint-dependent visibility function of all Gaussians in a trained model using a small, shared MLP across instances of an asset in a scene. By querying it for Gaussians within the viewing frustum prior to rasterization, our method can discard occluded primitives during rendering. Leveraging Tensor Cores for efficient computation, we integrate these neural queries directly into a novel instanced software rasterizer. Our approach outperforms the current state of the art for composed scenes in terms of VRAM usage and image quality, utilizing a combination of our instanced rasterizer and occlusion culling MLP, and exhibits complementary properties to existing LoD techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19172v1">MetroGS: Efficient and Stable Reconstruction of Geometrically Accurate High-Fidelity Large-Scale Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Project page: https://m3phist0.github.io/MetroGS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting and its derivatives have achieved significant breakthroughs in large-scale scene reconstruction. However, how to efficiently and stably achieve high-quality geometric fidelity remains a core challenge. To address this issue, we introduce MetroGS, a novel Gaussian Splatting framework for efficient and robust reconstruction in complex urban environments. Our method is built upon a distributed 2D Gaussian Splatting representation as the core foundation, serving as a unified backbone for subsequent modules. To handle potential sparse regions in complex scenes, we propose a structured dense enhancement scheme that utilizes SfM priors and a pointmap model to achieve a denser initialization, while incorporating a sparsity compensation mechanism to improve reconstruction completeness. Furthermore, we design a progressive hybrid geometric optimization strategy that organically integrates monocular and multi-view optimization to achieve efficient and accurate geometric refinement. Finally, to address the appearance inconsistency commonly observed in large-scale scenes, we introduce a depth-guided appearance modeling approach that learns spatial features with 3D consistency, facilitating effective decoupling between geometry and appearance and further enhancing reconstruction stability. Experiments on large-scale urban datasets demonstrate that MetroGS achieves superior geometric accuracy, rendering quality, offering a unified solution for high-fidelity large-scale scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.03121v2">Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has demonstrated impressive 3D reconstruction performance with explicit scene representations. Given the widespread application of 3DGS in 3D reconstruction and generation tasks, there is an urgent need to protect the copyright of 3DGS assets. However, existing copyright protection techniques for 3DGS overlook the usability of 3D assets, posing challenges for practical deployment. Here we describe splats in splats, the first 3DGS steganography framework that embeds 3D content in 3DGS itself without modifying any attributes. To achieve this, we take a deep insight into spherical harmonics (SH) and devise an importance-graded SH coefficient encryption strategy to embed the hidden SH coefficients. Furthermore, we employ a convolutional autoencoder to establish a mapping between the original Gaussian primitives' opacity and the hidden Gaussian primitives' opacity. Extensive experiments indicate that our method significantly outperforms existing 3D steganography techniques, with 5.31% higher scene fidelity and 3x faster rendering speed, while ensuring security, robustness, and user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16301v2">Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      We present \textbf{Upsample Anything}, a lightweight test-time optimization (TTO) framework that restores low-resolution features to high-resolution, pixel-wise outputs without any training. Although Vision Foundation Models demonstrate strong generalization across diverse downstream tasks, their representations are typically downsampled by 14x/16x (e.g., ViT), which limits their direct use in pixel-level applications. Existing feature upsampling approaches depend on dataset-specific retraining or heavy implicit optimization, restricting scalability and generalization. Upsample Anything addresses these issues through a simple per-image optimization that learns an anisotropic Gaussian kernel combining spatial and range cues, effectively bridging Gaussian Splatting and Joint Bilateral Upsampling. The learned kernel acts as a universal, edge-aware operator that transfers seamlessly across architectures and modalities, enabling precise high-resolution reconstruction of features, depth, or probability maps. It runs in only $\approx0.419 \text{s}$ per 224x224 image and achieves state-of-the-art performance on semantic segmentation, depth estimation, and both depth and probability map upsampling. \textbf{Project page:} \href{https://seominseok0429.github.io/Upsample-Anything/}{https://seominseok0429.github.io/Upsample-Anything/}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18873v1">Neural Texture Splatting: Expressive 3D Gaussian Splatting for View Synthesis, Geometry, and Dynamic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 SIGGRAPH Asia 2025 (conference track), Project page: https://19reborn.github.io/nts/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading approach for high-quality novel view synthesis, with numerous variants extending its applicability to a broad spectrum of 3D and 4D scene reconstruction tasks. Despite its success, the representational capacity of 3DGS remains limited by the use of 3D Gaussian kernels to model local variations. Recent works have proposed to augment 3DGS with additional per-primitive capacity, such as per-splat textures, to enhance its expressiveness. However, these per-splat texture approaches primarily target dense novel view synthesis with a reduced number of Gaussian primitives, and their effectiveness tends to diminish when applied to more general reconstruction scenarios. In this paper, we aim to achieve concrete performance improvement over state-of-the-art 3DGS variants across a wide range of reconstruction tasks, including novel view synthesis, geometry and dynamic reconstruction, under both sparse and dense input settings. To this end, we introduce Neural Texture Splatting (NTS). At the core of our approach is a global neural field (represented as a hybrid of a tri-plane and a neural decoder) that predicts local appearance and geometric fields for each primitive. By leveraging this shared global representation that models local texture fields across primitives, we significantly reduce model size and facilitate efficient global information exchange, demonstrating strong generalization across tasks. Furthermore, our neural modeling of local texture fields introduces expressive view- and time-dependent effects, a critical aspect that existing methods fail to account for. Extensive experiments show that Neural Texture Splatting consistently improves models and achieves state-of-the-art results across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17951v2">SplatCo: Structure-View Collaborative Gaussian Splatting for Detail-Preserving Rendering of Large-Scale Unbounded Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      We present SplatCo, a structure-view collaborative Gaussian splatting framework for high-fidelity rendering of complex outdoor environments. SplatCo builds upon two novel components: (1) a cross-structure collaboration module that combines global tri-plane representations, which capture coarse scene layouts, with local context grid features that represent fine surface details. This fusion is achieved through a novel hierarchical compensation strategy, ensuring both global consistency and local detail preservation; and (2) a cross-view assisted training strategy that enhances multi-view consistency by synchronizing gradient updates across viewpoints, applying visibility-aware densification, and pruning overfitted or inaccurate Gaussians based on structural consistency. Through joint optimization of structural representation and multi-view coherence, SplatCo effectively reconstructs fine-grained geometric structures and complex textures in large-scale scenes. Comprehensive evaluations on 13 diverse large-scale scenes, including Mill19, MatrixCity, Tanks & Temples, WHU, and custom aerial captures, demonstrate that SplatCo consistently achieves higher reconstruction quality than state-of-the-art methods, with PSNR improvements of 1-2 dB and SSIM gains of 0.1 to 0.2. These results establish a new benchmark for high-fidelity rendering of large-scale unbounded scenes. Code and additional information are available at https://github.com/SCUT-BIP-Lab/SplatCo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14477v2">2D Gaussians Spatial Transport for Point-supervised Density Regression</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 15 pages, 6 figures. This is the preprint version of the paper and supplemental material to appear in AAAI, 2026. Please cite the final published version. Code is available at https://github.com/infinite0522/GST
    </div>
    <details class="paper-abstract">
      This paper introduces Gaussian Spatial Transport (GST), a novel framework that leverages Gaussian splatting to facilitate transport from the probability measure in the image coordinate space to the annotation map. We propose a Gaussian splatting-based method to estimate pixel-annotation correspondence, which is then used to compute a transport plan derived from Bayesian probability. To integrate the resulting transport plan into standard network optimization in typical computer vision tasks, we derive a loss function that measures discrepancy after transport. Extensive experiments on representative computer vision tasks, including crowd counting and landmark detection, validate the effectiveness of our approach. Compared to conventional optimal transport schemes, GST eliminates iterative transport plan computation during training, significantly improving efficiency. Code is available at https://github.com/infinite0522/GST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17092v2">SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. Then we compress 3D Gaussians into planar Gaussians to facilitate accurate estimation of normal and depth. The planar Gaussians are optimized in a coarse-to-fine manner through depth smooth regularization and few-shot diffusion. Moreover, we introduce a part segmentation probability for each Gaussian primitive and update them by back-projecting part segmentation masks of renderings. Extensive experimental results demonstrate that our method achieves higher-fidelity part-level surface reconstruction on both synthetic and real-world data than existing methods. Codes will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17059v2">REArtGS++: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are pervasive in daily environments, such as drawers and refrigerators. Towards their part-level surface reconstruction and joint parameter estimation, REArtGS introduces a category-agnostic approach using multi-view RGB images at two different states. However, we observe that REArtGS still struggles with screw-joint or multi-part objects and lacks geometric constraints for unseen states. In this paper, we propose REArtGS++, a novel method towards generalizable articulated object reconstruction with temporal geometry constraint and planar Gaussian splatting. We first model a decoupled screw motion for each joint without type prior, and jointly optimize part-aware Gaussians with joint parameters through part motion blending. To introduce time-continuous geometric constraint for articulated modeling, we encourage Gaussians to be planar and propose a temporally consistent regularization between planar normal and depth through Taylor first-order expansion. Extensive experiments on both synthetic and real-world articulated objects demonstrate our superiority in generalizable part-level surface reconstruction and joint parameter estimation, compared to existing approaches. Project Site: https://sites.google.com/view/reartgs2/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18755v1">Splatonic: Architecture Support for 3D Gaussian Splatting SLAM via Sparse Processing</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has emerged as a promising direction for SLAM due to its high-fidelity reconstruction and rapid convergence. However, 3DGS-SLAM algorithms remain impractical for mobile platforms due to their high computational cost, especially for their tracking process. This work introduces Splatonic, a sparse and efficient real-time 3DGS-SLAM algorithm-hardware co-design for resource-constrained devices. Inspired by classical SLAMs, we propose an adaptive sparse pixel sampling algorithm that reduces the number of rendered pixels by up to 256$\times$ while retaining accuracy. To unlock this performance potential on mobile GPUs, we design a novel pixel-based rendering pipeline that improves hardware utilization via Gaussian-parallel rendering and preemptive $α$-checking. Together, these optimizations yield up to 121.7$\times$ speedup on the bottleneck stages and 14.6$\times$ end-to-end speedup on off-the-shelf GPUs. To further address new bottlenecks introduced by our rendering pipeline, we propose a pipelined architecture that simplifies the overall design while addressing newly emerged bottlenecks in projection and aggregation. Evaluated across four 3DGS-SLAM algorithms, Splatonic achieves up to 274.9$\times$ speedup and 4738.5$\times$ energy savings over mobile GPUs and up to 25.2$\times$ speedup and 241.1$\times$ energy savings over state-of-the-art accelerators, all with comparable accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.07608v3">Faster and Better 3D Splatting via Group Training</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to ICCV 2025. Code is available at https://github.com/Chengbo-Wang/3DGS-with-Group-Training
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, demonstrating remarkable capability in high-fidelity scene reconstruction through its Gaussian primitive representations. However, the computational overhead induced by the massive number of primitives poses a significant bottleneck to training efficiency. To overcome this challenge, we propose Group Training, a simple yet effective strategy that organizes Gaussian primitives into manageable groups, optimizing training efficiency and improving rendering quality. This approach shows universal compatibility with existing 3DGS frameworks, including vanilla 3DGS and Mip-Splatting, consistently achieving accelerated training while maintaining superior synthesis quality. Extensive experiments reveal that our straightforward Group Training strategy achieves up to 30\% faster convergence and improved rendering quality across diverse scenarios. Project Website: https://chengbo-wang.github.io/3DGS-with-Group-Training/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.14698v2">Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 NeurIPS 2025, Previously titled "SplatVoxel: History-Aware Novel View Streaming without Temporal Training", Project Page: https://19reborn.github.io/SplatVoxel/
    </div>
    <details class="paper-abstract">
      Recent advances in feed-forward 3D Gaussian Splatting have led to rapid improvements in efficient scene reconstruction from sparse views. However, most existing approaches construct Gaussian primitives directly aligned with the pixels in one or more of the input images. This leads to redundancies in the representation when input views overlap and constrains the position of the primitives to lie along the input rays without full flexibility in 3D space. Moreover, these pixel-aligned approaches do not naturally generalize to dynamic scenes, where effectively leveraging temporal information requires resolving both redundant and newly appearing content across frames. To address these limitations, we introduce a novel Fuse-and-Refine module that enhances existing feed-forward models by merging and refining the primitives in a canonical 3D space. At the core of our method is an efficient hybrid Splat-Voxel representation: from an initial set of pixel-aligned Gaussian primitives, we aggregate local features into a coarse-to-fine voxel hierarchy, and then use a sparse voxel transformer to process these voxel features and generate refined Gaussian primitives. By fusing and refining an arbitrary number of inputs into a consistent set of primitives, our representation effectively reduces redundancy and naturally adapts to temporal frames, enabling history-aware online reconstruction of dynamic scenes. Our approach achieves state-of-the-art performance in both static and streaming scene reconstructions while running at interactive rates (15 fps with 350ms delay) on a single H100 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19542v1">Proxy-Free Gaussian Splats Deformation with Splat-Based Surface Estimation</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 17 pages, Accepted to 3DV 2026 (IEEE/CVF International Conference on 3D Vision)
    </div>
    <details class="paper-abstract">
      We introduce SpLap, a proxy-free deformation method for Gaussian splats (GS) based on a Laplacian operator computed from our novel surface-aware splat graph. Existing approaches to GS deformation typically rely on deformation proxies such as cages or meshes, but they suffer from dependency on proxy quality and additional computational overhead. An alternative is to directly apply Laplacian-based deformation techniques by treating splats as point clouds. However, this often fail to properly capture surface information due to lack of explicit structure. To address this, we propose a novel method that constructs a surface-aware splat graph, enabling the Laplacian operator derived from it to support more plausible deformations that preserve details and topology. Our key idea is to leverage the spatial arrangement encoded in splats, defining neighboring splats not merely by the distance between their centers, but by their intersections. Furthermore, we introduce a Gaussian kernel adaptation technique that preserves surface structure under deformation, thereby improving rendering quality after deformation. In our experiments, we demonstrate the superior performance of our method compared to both proxy-based and proxy-free baselines, evaluated on 50 challenging objects from the ShapeNet, Objaverse, and Sketchfab datasets, as well as the NeRF-Synthetic dataset. Code is available at https://github.com/kjae0/SpLap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18570v1">PhysGS: Bayesian-Inferred Gaussian Splatting for Physical Property Estimation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Submitted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Understanding physical properties such as friction, stiffness, hardness, and material composition is essential for enabling robots to interact safely and effectively with their surroundings. However, existing 3D reconstruction methods focus on geometry and appearance and cannot infer these underlying physical properties. We present PhysGS, a Bayesian-inferred extension of 3D Gaussian Splatting that estimates dense, per-point physical properties from visual cues and vision--language priors. We formulate property estimation as Bayesian inference over Gaussian splats, where material and property beliefs are iteratively refined as new observations arrive. PhysGS also models aleatoric and epistemic uncertainties, enabling uncertainty-aware object and scene interpretation. Across object-scale (ABO-500), indoor, and outdoor real-world datasets, PhysGS improves accuracy of the mass estimation by up to 22.8%, reduces Shore hardness error by up to 61.2%, and lowers kinetic friction error by up to 18.1% compared to deterministic baselines. Our results demonstrate that PhysGS unifies 3D reconstruction, uncertainty modeling, and physical reasoning in a single, spatially continuous framework for dense physical property estimation. Additional results are available at https://samchopra2003.github.io/physgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18525v1">Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Submitted to ICRA 2026
    </div>
    <details class="paper-abstract">
      We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: https://splatblox.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18441v1">ReCoGS: Real-time ReColoring for Gaussian Splatting scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Project page is available at https://github.com/loryruta/recogs
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a leading method for novel view synthesis, offering superior training efficiency and real-time inference compared to NeRF approaches, while still delivering high-quality reconstructions. Beyond view synthesis, this 3D representation has also been explored for editing tasks. Many existing methods leverage 2D diffusion models to generate multi-view datasets for training, but they often suffer from limitations such as view inconsistencies, lack of fine-grained control, and high computational demand. In this work, we focus specifically on the editing task of recoloring. We introduce a user-friendly pipeline that enables precise selection and recoloring of regions within a pre-trained Gaussian Splatting scene. To demonstrate the real-time performance of our method, we also present an interactive tool that allows users to experiment with the pipeline in practice. Code is available at https://github.com/loryruta/recogs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18386v1">SegSplat: Feed-forward Gaussian Splatting and Open-Set Semantic Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      We have introduced SegSplat, a novel framework designed to bridge the gap between rapid, feed-forward 3D reconstruction and rich, open-vocabulary semantic understanding. By constructing a compact semantic memory bank from multi-view 2D foundation model features and predicting discrete semantic indices alongside geometric and appearance attributes for each 3D Gaussian in a single pass, SegSplat efficiently imbues scenes with queryable semantics. Our experiments demonstrate that SegSplat achieves geometric fidelity comparable to state-of-the-art feed-forward 3D Gaussian Splatting methods while simultaneously enabling robust open-set semantic segmentation, crucially \textit{without} requiring any per-scene optimization for semantic feature integration. This work represents a significant step towards practical, on-the-fly generation of semantically aware 3D environments, vital for advancing robotic interaction, augmented reality, and other intelligent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18367v1">Alias-free 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Project page: https://4d-alias-free.github.io/4D-Alias-free/
    </div>
    <details class="paper-abstract">
      Existing dynamic scene reconstruction methods based on Gaussian Splatting enable real-time rendering and generate realistic images. However, adjusting the camera's focal length or the distance between Gaussian primitives and the camera to modify rendering resolution often introduces strong artifacts, stemming from the frequency constraints of 4D Gaussians and Gaussian scale mismatch induced by the 2D dilated filter. To address this, we derive a maximum sampling frequency formulation for 4D Gaussian Splatting and introduce a 4D scale-adaptive filter and scale loss, which flexibly regulates the sampling frequency of 4D Gaussian Splatting. Our approach eliminates high-frequency artifacts under increased rendering frequencies while effectively reducing redundant Gaussians in multi-view video reconstruction. We validate the proposed method through monocular and multi-view video reconstruction experiments.Ours project page: https://4d-alias-free.github.io/4D-Alias-free/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10936v2">Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Accepted by AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18140v1">Observer Actor: Active Vision Imitation Learning with Sparse View Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Videos are available on our project webpage at https://obact.github.io
    </div>
    <details class="paper-abstract">
      We propose Observer Actor (ObAct), a novel framework for active vision imitation learning in which the observer moves to optimal visual observations for the actor. We study ObAct on a dual-arm robotic system equipped with wrist-mounted cameras. At test time, ObAct dynamically assigns observer and actor roles: the observer arm constructs a 3D Gaussian Splatting (3DGS) representation from three images, virtually explores this to find an optimal camera pose, then moves to this pose; the actor arm then executes a policy using the observer's observations. This formulation enhances the clarity and visibility of both the object and the gripper in the policy's observations. As a result, we enable the training of ambidextrous policies on observations that remain closer to the occlusion-free training distribution, leading to more robust policies. We study this formulation with two existing imitation learning methods -- trajectory transfer and behavior cloning -- and experiments show that ObAct significantly outperforms static-camera setups: trajectory transfer improves by 145% without occlusion and 233% with occlusion, while behavior cloning improves by 75% and 143%, respectively. Videos are available at https://obact.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13186v2">STT-GS: Sample-Then-Transmit Edge Gaussian Splatting with Joint Client Selection and Power Control</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Edge Gaussian splatting (EGS), which aggregates data from distributed clients and trains a global GS model at the edge server, is an emerging paradigm for scene reconstruction. Unlike traditional edge resource management methods that emphasize communication throughput or general-purpose learning performance, EGS explicitly aims to maximize the GS qualities, rendering existing approaches inapplicable. To address this problem, this paper formulates a novel GS-oriented objective function that distinguishes the heterogeneous view contributions of different clients. However, evaluating this function in turn requires clients' images, leading to a causality dilemma. To this end, this paper further proposes a sample-then-transmit EGS (or STT-GS for short) strategy, which first samples a subset of images as pilot data from each client for loss prediction. Based on the first-stage evaluation, communication resources are then prioritized towards more valuable clients. To achieve efficient sampling, a feature-domain clustering (FDC) scheme is proposed to select the most representative data and pilot transmission time minimization (PTTM) is adopted to reduce the pilot overhead.Subsequently, we develop a joint client selection and power control (JCSPC) framework to maximize the GS-oriented function under communication resource constraints. Despite the nonconvexity of the problem, we propose a low-complexity efficient solution based on the penalty alternating majorization minimization (PAMM) algorithm. Experiments unveil that the proposed scheme significantly outperforms existing benchmarks on real-world datasets. It is found that the GS-oriented objective can be accurately predicted with low sampling ratios (e.g.,10%), and our method achieves an excellent tradeoff between view contributions and communication costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06299v3">Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Accepted by AAAI-26
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS), an explicit scene representation technique, has shown significant promise for dynamic novel-view synthesis from monocular video input. However, purely data-driven 3DGS often struggles to capture the diverse physics-driven motion patterns in dynamic scenes. To fill this gap, we propose Physics-Informed Deformable Gaussian Splatting (PIDG), which treats each Gaussian particle as a Lagrangian material point with time-varying constitutive parameters and is supervised by 2D optical flow via motion projection. Specifically, we adopt static-dynamic decoupled 4D decomposed hash encoding to reconstruct geometry and motion efficiently. Subsequently, we impose the Cauchy momentum residual as a physics constraint, enabling independent prediction of each particle's velocity and constitutive stress via a time-evolving material field. Finally, we further supervise data fitting by matching Lagrangian particle flow to camera-compensated optical flow, which accelerates convergence and improves generalization. Experiments on a custom physics-driven dataset as well as on standard synthetic and real-world datasets demonstrate significant gains in physical consistency and monocular dynamic reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05859v2">D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 AAAI-26 accepted, code: https://github.com/Mr-Zwkid/D-FCGS
    </div>
    <details class="paper-abstract">
      Free-Viewpoint Video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representation remains a major challenge. Existing dynamic 3D Gaussian Splatting methods couple reconstruction with optimization-dependent compression and customized motion formats, limiting generalization and standardization. To address this, we propose D-FCGS, a novel Feedforward Compression framework for Dynamic Gaussian Splatting. Key innovations include: (1) a standardized Group-of-Frames (GoF) structure with I-P coding, leveraging sparse control points to extract inter-frame motion tensors; (2) a dual prior-aware entropy model that fuses hyperprior and spatial-temporal priors for accurate rate estimation; (3) a control-point-guided motion compensation mechanism and refinement network to enhance view-consistent fidelity. Trained on Gaussian frames derived from multi-view videos, D-FCGS generalizes across diverse scenes in a zero-shot fashion. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 40 times compression compared to the baseline while preserving visual quality across viewpoints. This work advances feedforward compression of dynamic 3DGS, facilitating scalable FVV transmission and storage for immersive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17932v1">Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Given just a few glimpses of a scene, can you imagine the movie playing out as the camera glides through it? That's the lens we take on \emph{sparse-input novel view synthesis}, not only as filling spatial gaps between widely spaced views, but also as \emph{completing a natural video} unfolding through space. We recast the task as \emph{test-time natural video completion}, using powerful priors from \emph{pretrained video diffusion models} to hallucinate plausible in-between views. Our \emph{zero-shot, generation-guided} framework produces pseudo views at novel camera poses, modulated by an \emph{uncertainty-aware mechanism} for spatial coherence. These synthesized frames densify supervision for \emph{3D Gaussian Splatting} (3D-GS) for scene reconstruction, especially in under-observed regions. An iterative feedback loop lets 3D geometry and 2D view synthesis inform each other, improving both the scene reconstruction and the generated views. The result is coherent, high-fidelity renderings from sparse inputs \emph{without any scene-specific training or fine-tuning}. On LLFF, DTU, DL3DV, and MipNeRF-360, our method significantly outperforms strong 3D-GS baselines under extreme sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17918v1">Frequency-Adaptive Sharpness Regularization for Improving 3D Gaussian Splatting Generalization</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Project page: https://bbangsik13.github.io/FASR
    </div>
    <details class="paper-abstract">
      Despite 3D Gaussian Splatting (3DGS) excelling in most configurations, it lacks generalization across novel viewpoints in a few-shot scenario because it overfits to the sparse observations. We revisit 3DGS optimization from a machine learning perspective, framing novel view synthesis as a generalization problem to unseen viewpoints-an underexplored direction. We propose Frequency-Adaptive Sharpness Regularization (FASR), which reformulates the 3DGS training objective, thereby guiding 3DGS to converge toward a better generalization solution. Although Sharpness-Aware Minimization (SAM) similarly reduces the sharpness of the loss landscape to improve generalization of classification models, directly employing it to 3DGS is suboptimal due to the discrepancy between the tasks. Specifically, it hinders reconstructing high-frequency details due to excessive regularization, while reducing its strength leads to under-penalizing sharpness. To address this, we reflect the local frequency of images to set the regularization weight and the neighborhood radius when estimating the local sharpness. It prevents floater artifacts in novel viewpoints and reconstructs fine details that SAM tends to oversmooth. Across datasets with various configurations, our method consistently improves a wide range of baselines. Code will be available at https://bbangsik13.github.io/FASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17904v1">CUS-GS: A Compact Unified Structured Gaussian Splatting Framework for Multimodal Scene Representation</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 15 pages, 8 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting based 3D scene representation have shown two major trends: semantics-oriented approaches that focus on high-level understanding but lack explicit 3D geometry modeling, and structure-oriented approaches that capture spatial structures yet provide limited semantic abstraction. To bridge this gap, we present CUS-GS, a compact unified structured Gaussian Splatting representation, which connects multimodal semantic features with structured 3D geometry. Specifically, we design a voxelized anchor structure that constructs a spatial scaffold, while extracting multimodal semantic features from a set of foundation models (e.g., CLIP, DINOv2, SEEM). Moreover, we introduce a multimodal latent feature allocation mechanism to unify appearance, geometry, and semantics across heterogeneous feature spaces, ensuring a consistent representation across multiple foundation models. Finally, we propose a feature-aware significance evaluation strategy to dynamically guide anchor growing and pruning, effectively removing redundant or invalid anchors while maintaining semantic integrity. Extensive experiments show that CUS-GS achieves competitive performance compared to state-of-the-art methods using as few as 6M parameters - an order of magnitude smaller than the closest rival at 35M - highlighting the excellent trade off between performance and model efficiency of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.19800v3">TrackGS: Optimizing COLMAP-Free 3D Gaussian Splatting with Global Track Constraints</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We present TrackGS, a novel method to integrate global feature tracks with 3D Gaussian Splatting (3DGS) for COLMAP-free novel view synthesis. While 3DGS delivers impressive rendering quality, its reliance on accurate precomputed camera parameters remains a significant limitation. Existing COLMAP-free approaches depend on local constraints that fail in complex scenarios. Our key innovation lies in leveraging feature tracks to establish global geometric constraints, enabling simultaneous optimization of camera parameters and 3D Gaussians. Specifically, we: (1) introduce track-constrained Gaussians that serve as geometric anchors, (2) propose novel 2D and 3D track losses to enforce multi-view consistency, and (3) derive differentiable formulations for camera intrinsics optimization. Extensive experiments on challenging real-world and synthetic datasets demonstrate state-of-the-art performance, with much lower pose error than previous methods while maintaining superior rendering quality. Our approach eliminates the need for COLMAP preprocessing, making 3DGS more accessible for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20714v2">Wideband RF Radiance Field Modeling Using Frequency-embedded 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Indoor environments typically contain diverse RF signals distributed across multiple frequency bands, including NB-IoT, Wi-Fi, and millimeter-wave. Consequently, wideband RF modeling is essential for practical applications such as joint deployment of heterogeneous RF systems, cross-band communication, and distributed RF sensing. Although 3D Gaussian Splatting (3DGS) techniques effectively reconstruct RF radiance fields at a single frequency, they cannot model fields at arbitrary or unknown frequencies across a wide range. In this paper, we present a novel 3DGS algorithm for unified wideband RF radiance field modeling. RF wave propagation depends on signal frequency and the 3D spatial environment, including geometry and material electromagnetic (EM) properties. To address these factors, we introduce a frequency-embedded EM feature network that utilizes 3D Gaussian spheres at each spatial location to learn the relationship between frequency and transmission characteristics, such as attenuation and radiance intensity. With a dataset containing sparse frequency samples in a specific 3D environment, our model can efficiently reconstruct RF radiance fields at arbitrary and unseen frequencies. To assess our approach, we introduce a large-scale power angular spectrum (PAS) dataset with 50,000 samples spanning 1 to 94 GHz across six indoor environments. Experimental results show that the proposed model trained on multiple frequencies achieves a Structural Similarity Index Measure (SSIM) of 0.922 for PAS reconstruction, surpassing state-of-the-art single-frequency 3DGS models with SSIM of 0.863.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17210v1">FisheyeGaussianLift: BEV Feature Lifting for Surround-View Fisheye Camera Perception</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 8 pages, 3 figures, published in IMVIP 2025 conference
    </div>
    <details class="paper-abstract">
      Accurate BEV semantic segmentation from fisheye imagery remains challenging due to extreme non-linear distortion, occlusion, and depth ambiguity inherent to wide-angle projections. We present a distortion-aware BEV segmentation framework that directly processes multi-camera high-resolution fisheye images,utilizing calibrated geometric unprojection and per-pixel depth distribution estimation. Each image pixel is lifted into 3D space via Gaussian parameterization, predicting spatial means and anisotropic covariances to explicitly model geometric uncertainty. The projected 3D Gaussians are fused into a BEV representation via differentiable splatting, producing continuous, uncertainty-aware semantic maps without requiring undistortion or perspective rectification. Extensive experiments demonstrate strong segmentation performance on complex parking and urban driving scenarios, achieving IoU scores of 87.75% for drivable regions and 57.26% for vehicles under severe fisheye distortion and diverse environmental conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17116v1">PEGS: Physics-Event Enhanced Large Spatiotemporal Motion Reconstruction via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Reconstruction of rigid motion over large spatiotemporal scales remains a challenging task due to limitations in modeling paradigms, severe motion blur, and insufficient physical consistency. In this work, we propose PEGS, a framework that integrates Physical priors with Event stream enhancement within a 3D Gaussian Splatting pipeline to perform deblurred target-focused modeling and motion recovery. We introduce a cohesive triple-level supervision scheme that enforces physical plausibility via an acceleration constraint, leverages event streams for high-temporal resolution guidance, and employs a Kalman regularizer to fuse multi-source observations. Furthermore, we design a motion-aware simulated annealing strategy that adaptively schedules the training process based on real-time kinematic states. We also contribute the first RGB-Event paired dataset targeting natural, fast rigid motion across diverse scenarios. Experiments show PEGS's superior performance in reconstructing motion over large spatiotemporal scales compared to mainstream dynamic methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17111v1">Towards Generative Design Using Optimal Transport for Shape Exploration and Solution Field Interpolation</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Generative Design (GD) combines artificial intelligence (AI), physics-based modeling, and multi-objective optimization to autonomously explore and refine engineering designs. Despite its promise in aerospace, automotive, and other high-performance applications, current GD methods face critical challenges: AI approaches require large datasets and often struggle to generalize; topology optimization is computationally intensive and difficult to extend to multiphysics problems; and model order reduction for evolving geometries remains underdeveloped. To address these challenges, we introduce a unified, structure-preserving framework for GD based on optimal transport (OT), enabling simultaneous interpolation of complex geometries and their associated physical solution fields across evolving design spaces, even with non-matching meshes and substantial shape changes. This capability leverages Gaussian splatting to provide a continuous, mesh-independent representation of the solution and Wasserstein barycenters to enable smooth, mathematically ''mass''-preserving blending of geometries, offering a major advance over surrogate models tied to static meshes. Our framework efficiently interpolates positive scalar fields across arbitrarily shaped, evolving geometries without requiring identical mesh topology or dimensionality. OT also naturally preserves localized physical features -- such as stress concentrations or sharp gradients -- by conserving the spatial distribution of quantities, interpreted as ''mass'' in a mathematical sense, rather than averaging them, avoiding artificial smoothing. Preliminary extensions to signed and vector fields are presented. Representative test cases demonstrate enhanced efficiency, adaptability, and physical fidelity, establishing a foundation for future foundation-model-powered generative design workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.06677v5">REArtGS: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 11pages, 6 figures
    </div>
    <details class="paper-abstract">
      Articulated objects, as prevalent entities in human life, their 3D representations play crucial roles across various applications. However, achieving both high-fidelity textured surface reconstruction and dynamic generation for articulated objects remains challenging for existing methods. In this paper, we present REArtGS, a novel framework that introduces additional geometric and motion constraints to 3D Gaussian primitives, enabling realistic surface reconstruction and generation for articulated objects. Specifically, given multi-view RGB images of arbitrary two states of articulated objects, we first introduce an unbiased Signed Distance Field (SDF) guidance to regularize Gaussian opacity fields, enhancing geometry constraints and improving surface reconstruction quality. Then we establish deformable fields for 3D Gaussians constrained by the kinematic structures of articulated objects, achieving unsupervised generation of surface meshes in unseen states. Extensive experiments on both synthetic and real datasets demonstrate our approach achieves high-quality textured surface reconstruction for given states, and enables high-fidelity surface generation for unseen states. Project site: https://sites.google.com/view/reartgs/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17092v1">SPAGS: Sparse-View Articulated Object Reconstruction from Single State via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous in daily environments, and their 3D reconstruction holds great significance across various fields. However, existing articulated object reconstruction methods typically require costly inputs such as multi-stage and multi-view observations. To address the limitations, we propose a category-agnostic articulated object reconstruction framework via planar Gaussian Splatting, which only uses sparse-view RGB images from a single state. Specifically, we first introduce a Gaussian information field to perceive the optimal sparse viewpoints from candidate camera poses. Then we compress 3D Gaussians into planar Gaussians to facilitate accurate estimation of normal and depth. The planar Gaussians are optimized in a coarse-to-fine manner through depth smooth regularization and few-shot diffusion. Moreover, we introduce a part segmentation probability for each Gaussian primitive and update them by back-projecting part segmentation masks of renderings. Extensive experimental results demonstrate that our method achieves higher-fidelity part-level surface reconstruction on both synthetic and real-world data than existing methods. Codes will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13278v2">SF-Recon: Simplification-Free Lightweight Building Reconstruction via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 This paper has been submitted to the 2026 ISPRS Congress
    </div>
    <details class="paper-abstract">
      Lightweight building surface models are crucial for digital city, navigation, and fast geospatial analytics, yet conventional multi-view geometry pipelines remain cumbersome and quality-sensitive due to their reliance on dense reconstruction, meshing, and subsequent simplification. This work presents SF-Recon, a method that directly reconstructs lightweight building surfaces from multi-view images without post-hoc mesh simplification. We first train an initial 3D Gaussian Splatting (3DGS) field to obtain a view-consistent representation. Building structure is then distilled by a normal-gradient-guided Gaussian optimization that selects primitives aligned with roof and wall boundaries, followed by multi-view edge-consistency pruning to enhance structural sharpness and suppress non-structural artifacts without external supervision. Finally, a multi-view depth-constrained Delaunay triangulation converts the structured Gaussian field into a lightweight, structurally faithful building mesh. Based on a proposed SF dataset, the experimental results demonstrate that our SF-Recon can directly reconstruct lightweight building models from multi-view imagery, achieving substantially fewer faces and vertices while maintaining computational efficiency. Website:https://lzh282140127-cell.github.io/SF-Recon-project/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17059v1">REArtGS++: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Articulated objects are pervasive in daily environments, such as drawers and refrigerators. Towards their part-level surface reconstruction and joint parameter estimation, REArtGS~\cite{wu2025reartgs} introduces a category-agnostic approach using multi-view RGB images at two different states. However, we observe that REArtGS still struggles with screw-joint or multi-part objects and lacks geometric constraints for unseen states. In this paper, we propose REArtGS++, a novel method towards generalizable articulated object reconstruction with temporal geometry constraint and planar Gaussian splatting. We first model a decoupled screw motion for each joint without type prior, and jointly optimize part-aware Gaussians with joint parameters through part motion blending. To introduce time-continuous geometric constraint for articulated modeling, we encourage Gaussians to be planar and propose a temporally consistent regularization between planar normal and depth through Taylor first-order expansion. Extensive experiments on both synthetic and real-world articulated objects demonstrate our superiority in generalizable part-level surface reconstruction and joint parameter estimation, compared to existing approaches. Project Site: https://sites.google.com/view/reartgs2/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16988v1">PhysMorph-GS: Differentiable Shape Morphing via Joint Optimization of Physics and Rendering Objectives</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 14pages, 12figures
    </div>
    <details class="paper-abstract">
      Shape morphing with physics-based simulation naturally supports large deformations and topology changes, but existing methods suffer from a "rendering gap": nondifferentiable surface extraction prevents image losses from directly guiding physics optimization. We introduce PhysMorph-GS, which couples a differentiable material point method (MPM) with 3D Gaussian splatting through a deformation-aware upsampling bridge that maps sparse particle states (x, F) to dense Gaussians (mu, Sigma). Multi-modal rendering losses on silhouette and depth backpropagate along two paths, from covariances to deformation gradients via a stretch-based mapping and from Gaussian means to particle positions. Through the MPM adjoint, these gradients update deformation controls while mass is conserved at a compact set of anchor particles. A multi-pass interleaved optimization scheme repeatedly injects rendering gradients into successive physics steps, avoiding collapse to purely physics-driven solutions. On challenging morphing sequences, PhysMorph-GS improves boundary fidelity and temporal stability over a differentiable MPM baseline and better reconstructs thin structures such as ears and tails. Quantitatively, our depth-supervised variant reduces Chamfer distance by about 2.5 percent relative to the physics-only baseline. By providing a differentiable particle-to-Gaussian bridge, PhysMorph-GS closes a key gap in physics-aware rendering pipelines and enables inverse design directly from image-space supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16980v1">Gradient-Driven Natural Selection for Compact 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      3DGS employs a large number of Gaussian primitives to fit scenes, resulting in substantial storage and computational overhead. Existing pruning methods rely on manually designed criteria or introduce additional learnable parameters, yielding suboptimal results. To address this, we propose an natural selection inspired pruning framework that models survival pressure as a regularization gradient field applied to opacity, allowing the optimization gradients--driven by the goal of maximizing rendering quality--to autonomously determine which Gaussians to retain or prune. This process is fully learnable and requires no human intervention. We further introduce an opacity decay technique with a finite opacity prior, which accelerates the selection process without compromising pruning effectiveness. Compared to 3DGS, our method achieves over 0.6 dB PSNR gain under 15\% budgets, establishing state-of-the-art performance for compact 3DGS. Project page https://xiaobin2001.github.io/GNS-web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21502v2">Generalizable and Relightable Gaussian Splatting for Human Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Project Webpage: https://sypj-98.github.io/grgs/
    </div>
    <details class="paper-abstract">
      We propose GRGS, a generalizable and relightable 3D Gaussian framework for high-fidelity human novel view synthesis under diverse lighting conditions. Unlike existing methods that rely on per-character optimization or ignore physical constraints, GRGS adopts a feed-forward, fully supervised strategy projecting geometry, material, and illumination cues from multi-view 2D observations into 3D Gaussian representations. To recover accurate geometry under diverse lighting conditions, we introduce a Lighting-robust Geometry Refinement (LGR) module trained on synthetically relit data to predict precise depth and surface normals. Based on the high-quality geometry, a Physically Grounded Neural Rendering (PGNR) module is further proposed to integrate neural prediction with physics-based shading, supporting editable relighting with shadows and indirect illumination. Moreover, we design a 2D-to-3D projection training scheme leveraging differentiable supervision from ambient occlusion, direct, and indirect lighting maps, alleviating the computational cost of ray tracing. Extensive experiments demonstrate that GRGS achieves superior visual quality, geometric consistency, and generalization across characters and lighting conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16966v1">One Walk is All You Need: Data-Efficient 3D RF Scene Reconstruction with Human Movements</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Reconstructing 3D Radiance Field (RF) scenes through opaque obstacles is a long-standing goal, yet it is fundamentally constrained by a laborious data acquisition process requiring thousands of static measurements, which treats human motion as noise to be filtered. This work introduces a new paradigm with a core objective: to perform fast, data-efficient, and high-fidelity RF reconstruction of occluded 3D static scenes, using only a single, brief human walk. We argue that this unstructured motion is not noise, but is in fact an information-rich signal available for reconstruction. To achieve this, we design a factorization framework based on composite 3D Gaussian Splatting (3DGS) that learns to model the dynamic effects of human motion from the persistent static scene geometry within a raw RF stream. Trained on just a single 60-second casual walk, our model reconstructs the full static scene with a Structural Similarity Index (SSIM) of 0.96, remarkably outperforming heavily-sampled state-of-the-art (SOTA) by 12%. By transforming the human movements into its valuable signals, our method eliminates the data acquisition bottleneck and paves the way for on-the-fly 3D RF mapping of unseen environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.01383v3">FalconWing: An Ultra-Light Indoor Fixed-Wing UAV Platform for Vision-Based Autonomy</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We introduce FalconWing, an ultra-light (150 g) indoor fixed-wing UAV platform for vision-based autonomy. Controlled indoor environment enables year-round repeatable UAV experiment but imposes strict weight and maneuverability limits on the UAV, motivating our ultra-light FalconWing design. FalconWing couples a lightweight hardware stack (137g airframe with a 9g camera) and offboard computation with a software stack featuring a photorealistic 3D Gaussian Splat (GSplat) simulator for developing and evaluating vision-based controllers. We validate FalconWing on two challenging vision-based aerial case studies. In the leader-follower case study, our best vision-based controller, trained via imitation learning on GSplat-rendered data augmented with domain randomization, achieves 100% tracking success across 3 types of leader maneuvers over 30 trials and shows robustness to leader's appearance shifts in simulation. In the autonomous landing case study, our vision-based controller trained purely in simulation transfers zero-shot to real hardware, achieving an 80% success rate over ten landing trials. We will release hardware designs, GSplat scenes, and dynamics models upon publication to make FalconWing an open-source flight kit for engineering students and research labs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17747v1">AEGIS: Preserving privacy of 3D Facial Avatars with Adversarial Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The growing adoption of photorealistic 3D facial avatars, particularly those utilizing efficient 3D Gaussian Splatting representations, introduces new risks of online identity theft, especially in systems that rely on biometric authentication. While effective adversarial masking methods have been developed for 2D images, a significant gap remains in achieving robust, viewpoint-consistent identity protection for dynamic 3D avatars. To address this, we present AEGIS, the first privacy-preserving identity masking framework for 3D Gaussian Avatars that maintains the subject's perceived characteristics. Our method aims to conceal identity-related facial features while preserving the avatar's perceptual realism and functional integrity. AEGIS applies adversarial perturbations to the Gaussian color coefficients, guided by a pre-trained face verification network, ensuring consistent protection across multiple viewpoints without retraining or modifying the avatar's geometry. AEGIS achieves complete de-identification, reducing face retrieval and verification accuracy to 0%, while maintaining high perceptual quality (SSIM = 0.9555, PSNR = 35.52 dB). It also preserves key facial attributes such as age, race, gender, and emotion, demonstrating strong privacy protection with minimal visual distortion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16542v1">EOGS++: Earth Observation Gaussian Splatting with Internal Camera Refinement and Direct Panchromatic Rendering</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 8 pages, ISPRS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting has been introduced as a compelling alternative to NeRF for Earth observation, offering com- petitive reconstruction quality with significantly reduced training times. In this work, we extend the Earth Observation Gaussian Splatting (EOGS) framework to propose EOGS++, a novel method tailored for satellite imagery that directly operates on raw high-resolution panchromatic data without requiring external preprocessing. Furthermore, leveraging optical flow techniques we embed bundle adjustment directly within the training process, avoiding reliance on external optimization tools while improving camera pose estimation. We also introduce several improvements to the original implementation, including early stopping and TSDF post-processing, all contributing to sharper reconstructions and better geometric accuracy. Experiments on the IARPA 2016 and DFC2019 datasets demonstrate that EOGS++ achieves state-of-the-art performance in terms of reconstruction quality and effi- ciency, outperforming the original EOGS method and other NeRF-based methods while maintaining the computational advantages of Gaussian Splatting. Our model demonstrates an improvement from 1.33 to 1.19 mean MAE errors on buildings compared to the original EOGS models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16301v1">Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      We present \textbf{Upsample Anything}, a lightweight test-time optimization (TTO) framework that restores low-resolution features to high-resolution, pixel-wise outputs without any training. Although Vision Foundation Models demonstrate strong generalization across diverse downstream tasks, their representations are typically downsampled by 14x/16x (e.g., ViT), which limits their direct use in pixel-level applications. Existing feature upsampling approaches depend on dataset-specific retraining or heavy implicit optimization, restricting scalability and generalization. Upsample Anything addresses these issues through a simple per-image optimization that learns an anisotropic Gaussian kernel combining spatial and range cues, effectively bridging Gaussian Splatting and Joint Bilateral Upsampling. The learned kernel acts as a universal, edge-aware operator that transfers seamlessly across architectures and modalities, enabling precise high-resolution reconstruction of features, depth, or probability maps. It runs in only $\approx0.419 \text{s}$ per 224x224 image and achieves state-of-the-art performance on semantic segmentation, depth estimation, and both depth and probability map upsampling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16298v1">Optimizing 3D Gaussian Splattering for Mobile GPUs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Image-based 3D scene reconstruction, which transforms multi-view images into a structured 3D representation of the surrounding environment, is a common task across many modern applications. 3D Gaussian Splatting (3DGS) is a new paradigm to address this problem and offers considerable efficiency as compared to the previous methods. Motivated by this, and considering various benefits of mobile device deployment (data privacy, operating without internet connectivity, and potentially faster responses), this paper develops Texture3dgs, an optimized mapping of 3DGS for a mobile GPU. A critical challenge in this area turns out to be optimizing for the two-dimensional (2D) texture cache, which needs to be exploited for faster executions on mobile GPUs. As a sorting method dominates the computations in 3DGS on mobile platforms, the core of Texture3dgs is a novel sorting algorithm where the processing, data movement, and placement are highly optimized for 2D memory. The properties of this algorithm are analyzed in view of a cost model for the texture cache. In addition, we accelerate other steps of the 3DGS algorithm through improved variable layout design and other optimizations. End-to-end evaluation shows that Texture3dgs delivers up to 4.1$\times$ and 1.7$\times$ speedup for the sorting and overall 3D scene reconstruction, respectively -- while also reducing memory usage by up to 1.6$\times$ -- demonstrating the effectiveness of our design for efficient mobile 3D scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16144v1">LEGO-SLAM: Language-Embedded Gaussian Optimization SLAM</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled Simultaneous Localization and Mapping (SLAM) systems to build photorealistic maps. However, these maps lack the open-vocabulary semantic understanding required for advanced robotic interaction. Integrating language features into SLAM remains a significant challenge, as storing high-dimensional features demands excessive memory and rendering overhead, while existing methods with static models lack adaptability for novel environments. To address these limitations, we propose LEGO-SLAM (Language-Embedded Gaussian Optimization SLAM), the first framework to achieve real-time, open-vocabulary mapping within a 3DGS-based SLAM system. At the core of our method is a scene-adaptive encoder-decoder that distills high-dimensional language embeddings into a compact 16-dimensional feature space. This design reduces the memory per Gaussian and accelerates rendering, enabling real-time performance. Unlike static approaches, our encoder adapts online to unseen scenes. These compact features also enable a language-guided pruning strategy that identifies semantic redundancy, reducing the map's Gaussian count by over 60\% while maintaining rendering quality. Furthermore, we introduce a language-based loop detection approach that reuses these mapping features, eliminating the need for a separate detection model. Extensive experiments demonstrate that LEGO-SLAM achieves competitive mapping quality and tracking accuracy, all while providing open-vocabulary capabilities at 15 FPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16112v1">Clustered Error Correction with Grouped 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 16 pages, 8 figures, SIGGRAPH Asia Conference Papers 2025
    </div>
    <details class="paper-abstract">
      Existing 4D Gaussian Splatting (4DGS) methods struggle to accurately reconstruct dynamic scenes, often failing to resolve ambiguous pixel correspondences and inadequate densification in dynamic regions. We address these issues by introducing a novel method composed of two key components: (1) Elliptical Error Clustering and Error Correcting Splat Addition that pinpoints dynamic areas to improve and initialize fitting splats, and (2) Grouped 4D Gaussian Splatting that improves consistency of mapping between splats and represented dynamic objects. Specifically, we classify rendering errors into missing-color and occlusion types, then apply targeted corrections via backprojection or foreground splitting guided by cross-view color consistency. Evaluations on Neural 3D Video and Technicolor datasets demonstrate that our approach significantly improves temporal consistency and achieves state-of-the-art perceptual rendering quality, improving 0.39dB of PSNR on the Technicolor Light Field dataset. Our visualization shows improved alignment between splats and dynamic objects, and the error correction method's capability to identify errors and properly initialize new splats. Our implementation details and source code are available at https://github.com/tho-kn/cem-4dgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16091v1">Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      We present Rad-GS, a 4D radar-camera SLAM system designed for kilometer-scale outdoor environments, utilizing 3D Gaussian as a differentiable spatial representation. Rad-GS combines the advantages of raw radar point cloud with Doppler information and geometrically enhanced point cloud to guide dynamic object masking in synchronized images, thereby alleviating rendering artifacts and improving localization accuracy. Additionally, unsynchronized image frames are leveraged to globally refine the 3D Gaussian representation, enhancing texture consistency and novel view synthesis fidelity. Furthermore, the global octree structure coupled with a targeted Gaussian primitive management strategy further suppresses noise and significantly reduces memory consumption in large-scale environments. Extensive experiments and ablation studies demonstrate that Rad-GS achieves performance comparable to traditional 3D Gaussian methods based on camera or LiDAR inputs, highlighting the feasibility of robust outdoor mapping using 4D mmWave radar. Real-world reconstruction at kilometer scale validates the potential of Rad-GS for large-scale scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16030v1">CuriGS: Curriculum-Guided Gaussian Splatting for Sparse View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as an efficient, high-fidelity representation for real-time scene reconstruction and rendering. However, extending 3DGS to sparse-view settings remains challenging because of supervision scarcity and overfitting caused by limited viewpoint coverage. In this paper, we present CuriGS, a curriculum-guided framework for sparse-view 3D reconstruction using 3DGS. CuriGS addresses the core challenge of sparse-view synthesis by introducing student views: pseudo-views sampled around ground-truth poses (teacher). For each teacher, we generate multiple groups of student views with different perturbation levels. During training, we follow a curriculum schedule that gradually unlocks higher perturbation level, randomly sampling candidate students from the active level to assist training. Each sampled student is regularized via depth-correlation and co-regularization, and evaluated using a multi-signal metric that combines SSIM, LPIPS, and an image-quality measure. For every teacher and perturbation level, we periodically retain the best-performing students and promote those that satisfy a predefined quality threshold to the training set, resulting in a stable augmentation of sparse training views. Experimental results show that CuriGS outperforms state-of-the-art baselines in both rendering fidelity and geometric consistency across various synthetic and real sparse-view scenes. Project page: https://zijian1026.github.io/CuriGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07917v3">SpeeDe3DGS: Speedy Deformable 3D Gaussian Splatting with Temporal Pruning and Motion Grouping</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Project Page: https://speede3dgs.github.io/
    </div>
    <details class="paper-abstract">
      Dynamic extensions of 3D Gaussian Splatting (3DGS) achieve high-quality reconstructions through neural motion fields, but per-Gaussian neural inference makes these models computationally expensive. Building on DeformableGS, we introduce Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), which bridges this efficiency-fidelity gap through three complementary modules: Temporal Sensitivity Pruning (TSP) removes low-impact Gaussians via temporally aggregated sensitivity analysis, Temporal Sensitivity Sampling (TSS) perturbs timestamps to suppress floaters and improve temporal coherence, and GroupFlow distills the learned deformation field into shared SE(3) transformations for efficient groupwise motion. On the 50 dynamic scenes in MonoDyGauBench, integrating TSP and TSS into DeformableGS accelerates rendering by 6.78$\times$ on average while maintaining neural-field fidelity and using 10$\times$ fewer primitives. Adding GroupFlow culminates in 13.71$\times$ faster rendering and 2.53$\times$ shorter training, surpassing all baselines in speed while preserving superior image quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16831v1">Vorion: A RISC-V GPU with Hardware-Accelerated 3D Gaussian Rendering and Training</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a foundational technique for real-time neural rendering, 3D scene generation, volumetric video (4D) capture. However, its rendering and training impose massive computation, making real-time rendering on edge devices and real-time 4D reconstruction on workstations currently infeasible. Given its fixed-function nature and similarity with traditional rasterization, 3DGS presents a strong case for dedicated hardware in the graphics pipeline of next-generation GPUs. This work, Vorion, presents the first GPGPU prototype with hardware-accelerated 3DGS rendering and training. Vorion features scalable architecture, minimal hardware change to traditional rasterizers, z-tiling to increase parallelism, and Gaussian/pixel-centric hybrid dataflow. We prototype the minimal system (8 SIMT cores, 2 Gaussian rasterizer) using TSMC 16nm FinFET technology, which achieves 19 FPS for rendering. The scaled design with 16 rasterizers achieves 38.6 iterations/s for training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.16898v4">MonoGSDF: Exploring Monocular Geometric Cues for Gaussian Splatting-Guided Implicit Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Accurate meshing from monocular images remains a key challenge in 3D vision. While state-of-the-art 3D Gaussian Splatting (3DGS) methods excel at synthesizing photorealistic novel views through rasterization-based rendering, their reliance on sparse, explicit primitives severely limits their ability to recover watertight and topologically consistent 3D surfaces.We introduce MonoGSDF, a novel method that couples Gaussian-based primitives with a neural Signed Distance Field (SDF) for high-quality reconstruction. During training, the SDF guides Gaussians' spatial distribution, while at inference, Gaussians serve as priors to reconstruct surfaces, eliminating the need for memory-intensive Marching Cubes. To handle arbitrary-scale scenes, we propose a scaling strategy for robust generalization. A multi-resolution training scheme further refines details and monocular geometric cues from off-the-shelf estimators enhance reconstruction quality. Experiments on real-world datasets show MonoGSDF outperforms prior methods while maintaining efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13264v2">SymGS : Leveraging Local Symmetries for 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Project Page: https://symgs.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as a transformative technique in novel view synthesis, primarily due to its high rendering speed and photorealistic fidelity. However, its memory footprint scales rapidly with scene complexity, often reaching several gigabytes. Existing methods address this issue by introducing compression strategies that exploit primitive-level redundancy through similarity detection and quantization. We aim to surpass the compression limits of such methods by incorporating symmetry-aware techniques, specifically targeting mirror symmetries to eliminate redundant primitives. We propose a novel compression framework, SymGS, introducing learnable mirrors into the scene, thereby eliminating local and global reflective redundancies for compression. Our framework functions as a plug-and-play enhancement to state-of-the-art compression methods, (e.g. HAC) to achieve further compression. Compared to HAC, we achieve $1.66 \times$ compression across benchmark datasets (upto $3\times$ on large-scale scenes). On an average, SymGS enables $\bf{108\times}$ compression of a 3DGS scene, while preserving rendering quality. The project page and supplementary can be found at symgs.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14270v2">Gaussian Splatting-based Low-Rank Tensor Representation for Multi-Dimensional Image Recovery</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Tensor singular value decomposition (t-SVD) is a promising tool for multi-dimensional image representation, which decomposes a multi-dimensional image into a latent tensor and an accompanying transform matrix. However, two critical limitations of t-SVD methods persist: (1) the approximation of the latent tensor (e.g., tensor factorizations) is coarse and fails to accurately capture spatial local high-frequency information; (2) The transform matrix is composed of fixed basis atoms (e.g., complex exponential atoms in DFT and cosine atoms in DCT) and cannot precisely capture local high-frequency information along the mode-3 fibers. To address these two limitations, we propose a Gaussian Splatting-based Low-rank tensor Representation (GSLR) framework, which compactly and continuously represents multi-dimensional images. Specifically, we leverage tailored 2D Gaussian splatting and 1D Gaussian splatting to generate the latent tensor and transform matrix, respectively. The 2D and 1D Gaussian splatting are indispensable and complementary under this representation framework, which enjoys a powerful representation capability, especially for local high-frequency information. To evaluate the representation ability of the proposed GSLR, we develop an unsupervised GSLR-based multi-dimensional image recovery model. Extensive experiments on multi-dimensional image recovery demonstrate that GSLR consistently outperforms state-of-the-art methods, particularly in capturing local high-frequency information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06909v2">Gaussian Mapping for Evolving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Mapping systems with novel view synthesis (NVS) capabilities, most notably 3D Gaussian Splatting (3DGS), are widely used in computer vision and across various applications, including augmented reality, robotics, and autonomous driving. However, many current approaches are limited to static scenes. While recent works have begun addressing short-term dynamics (motion within the camera's view), long-term dynamics (the scene evolving through changes out of view) remain less explored. To overcome this limitation, we introduce a dynamic scene-adaptation mechanism that continuously updates 3DGS to reflect the latest changes. Since maintaining consistency remains challenging due to stale observations that disrupt the reconstruction process, we propose a novel keyframe management mechanism that discards outdated observations while preserving as much information as possible. We thoroughly evaluate Gaussian Mapping for Evolving Scenes (\ours) on both synthetic and real-world datasets, achieving a 29.7\% improvement in PSNR and a 3 times improvement in L1 depth error over the most competitive baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14161v2">RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15102v1">Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      The recent introduction of 3D Gaussian Splatting (3DGS) has significantly advanced novel view synthesis. Several studies have further improved the rendering quality of 3DGS, yet they still exhibit noticeable visual discrepancies when synthesizing views at sampling rates unseen during training. Specifically, they suffer from (i) erosion-induced blurring artifacts when zooming in and (ii) dilation-induced staircase artifacts when zooming out. We speculate that these artifacts arise from the fundamental limitation of the alpha blending adopted in 3DGS methods. Instead of the conventional alpha blending that computes alpha and transmittance as scalar quantities over a pixel, we propose to replace it with our novel Gaussian Blending that treats alpha and transmittance as spatially varying distributions. Thus, transmittances can be updated considering the spatial distribution of alpha values across the pixel area, allowing nearby background splats to contribute to the final rendering. Our Gaussian Blending maintains real-time rendering speed and requires no additional memory cost, while being easily integrated as a drop-in replacement into existing 3DGS-based or other NVS frameworks. Extensive experiments demonstrate that Gaussian Blending effectively captures fine details at various sampling rates unseen during training, consistently outperforming existing novel view synthesis models across both unseen and seen sampling rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16467v2">Arbitrary-Scale 3D Gaussian Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting (3DGS) super-resolution methods typically perform high-resolution (HR) rendering of fixed scale factors, making them impractical for resource-limited scenarios. Directly rendering arbitrary-scale HR views with vanilla 3DGS introduces aliasing artifacts due to the lack of scale-aware rendering ability, while adding a post-processing upsampler for 3DGS complicates the framework and reduces rendering efficiency. To tackle these issues, we build an integrated framework that incorporates scale-aware rendering, generative prior-guided optimization, and progressive super-resolving to enable 3D Gaussian super-resolution of arbitrary scale factors with a single 3D model. Notably, our approach supports both integer and non-integer scale rendering to provide more flexibility. Extensive experiments demonstrate the effectiveness of our model in rendering high-quality arbitrary-scale HR views (6.59 dB PSNR gain over 3DGS) with a single model. It preserves structural consistency with LR views and across different scales, while maintaining real-time rendering speed (85 FPS at 1080p).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14633v1">SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted at AAAI 2026. Project page: https://miya-oi.github.io/SparseSurf-project
    </div>
    <details class="paper-abstract">
      Recent advances in optimizing Gaussian Splatting for scene geometry have enabled efficient reconstruction of detailed surfaces from images. However, when input views are sparse, such optimization is prone to overfitting, leading to suboptimal reconstruction quality. Existing approaches address this challenge by employing flattened Gaussian primitives to better fit surface geometry, combined with depth regularization to alleviate geometric ambiguities under limited viewpoints. Nevertheless, the increased anisotropy inherent in flattened Gaussians exacerbates overfitting in sparse-view scenarios, hindering accurate surface fitting and degrading novel view synthesis performance. In this paper, we propose \net{}, a method that reconstructs more accurate and detailed surfaces while preserving high-quality novel view rendering. Our key insight is to introduce Stereo Geometry-Texture Alignment, which bridges rendering quality and geometry estimation, thereby jointly enhancing both surface reconstruction and view synthesis. In addition, we present a Pseudo-Feature Enhanced Geometry Consistency that enforces multi-view geometric consistency by incorporating both training and unseen views, effectively mitigating overfitting caused by sparse supervision. Extensive experiments on the DTU, BlendedMVS, and Mip-NeRF360 datasets demonstrate that our method achieves the state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14540v1">Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      This paper focuses on a challenging setting of simultaneously modeling geometry and appearance of hand-object interaction scenes without any object priors. We follow the trend of dynamic 3D Gaussian Splatting based methods, and address several significant challenges. To model complex hand-object interaction with mutual occlusion and edge blur, we present interaction-aware hand-object Gaussians with newly introduced optimizable parameters aiming to adopt piecewise linear hypothesis for clearer structural representation. Moreover, considering the complementarity and tightness of hand shape and object shape during interaction dynamics, we incorporate hand information into object deformation field, constructing interaction-aware dynamic fields to model flexible motions. To further address difficulties in the optimization process, we propose a progressive strategy that handles dynamic regions and static background step by step. Correspondingly, explicit regularizations are designed to stabilize the hand-object representations for smooth motion transition, physical interaction reality, and coherent lighting. Experiments show that our approach surpasses existing dynamic 3D-GS-based methods and achieves state-of-the-art performance in reconstructing dynamic hand-object interaction.
    </details>
</div>
