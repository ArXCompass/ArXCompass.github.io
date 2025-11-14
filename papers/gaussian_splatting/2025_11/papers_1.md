# gaussian splatting - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07743v1">UltraGS: Gaussian Splatting for Ultrasound Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view complicates novel view synthesis. We propose \textbf{UltraGS}, a Gaussian Splatting framework optimized for ultrasound imaging. First, we introduce a depth-aware Gaussian splatting strategy, where each Gaussian is assigned a learnable field of view, enabling accurate depth prediction and precise structural representation. Second, we design SH-DARS, a lightweight rendering function combining low-order spherical harmonics with ultrasound-specific wave physics, including depth attenuation, reflection, and scattering, to model tissue intensity accurately. Third, we contribute the Clinical Ultrasound Examination Dataset, a benchmark capturing diverse anatomical scans under real-world clinical protocols. Extensive experiments on three datasets demonstrate UltraGS's superiority, achieving state-of-the-art results in PSNR (up to 29.55), SSIM (up to 0.89), and MSE (as low as 0.002) while enabling real-time synthesis at 64.69 fps. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07743v1">UltraGS: Gaussian Splatting for Ultrasound Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ultrasound imaging is a cornerstone of non-invasive clinical diagnostics, yet its limited field of view complicates novel view synthesis. We propose \textbf{UltraGS}, a Gaussian Splatting framework optimized for ultrasound imaging. First, we introduce a depth-aware Gaussian splatting strategy, where each Gaussian is assigned a learnable field of view, enabling accurate depth prediction and precise structural representation. Second, we design SH-DARS, a lightweight rendering function combining low-order spherical harmonics with ultrasound-specific wave physics, including depth attenuation, reflection, and scattering, to model tissue intensity accurately. Third, we contribute the Clinical Ultrasound Examination Dataset, a benchmark capturing diverse anatomical scans under real-world clinical protocols. Extensive experiments on three datasets demonstrate UltraGS's superiority, achieving state-of-the-art results in PSNR (up to 29.55), SSIM (up to 0.89), and MSE (as low as 0.002) while enabling real-time synthesis at 64.69 fps. The code and dataset are open-sourced at: https://github.com/Bean-Young/UltraGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2505.21890v2">Diffusion Denoised Hyperspectral Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to 3DV 2026
    </div>
    <details class="paper-abstract">
      Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements of samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can be used to render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of hyperspectral scenes across the full spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of DD-HGS. The results demonstrate that DD-HGS achieves new state-of-the-art performance among previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05229v1">4D3R: Motion-Aware Neural Reconstruction and Rendering of Dynamic Scenes from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Novel view synthesis from monocular videos of dynamic scenes with unknown camera poses remains a fundamental challenge in computer vision and graphics. While recent advances in 3D representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown promising results for static scenes, they struggle with dynamic content and typically rely on pre-computed camera poses. We present 4D3R, a pose-free dynamic neural rendering framework that decouples static and dynamic components through a two-stage approach. Our method first leverages 3D foundational models for initial pose and geometry estimation, followed by motion-aware refinement. 4D3R introduces two key technical innovations: (1) a motion-aware bundle adjustment (MA-BA) module that combines transformer-based learned priors with SAM2 for robust dynamic object segmentation, enabling more accurate camera pose refinement; and (2) an efficient Motion-Aware Gaussian Splatting (MA-GS) representation that uses control points with a deformation field MLP and linear blend skinning to model dynamic motion, significantly reducing computational cost while maintaining high-quality reconstruction. Extensive experiments on real-world dynamic datasets demonstrate that our approach achieves up to 1.8dB PSNR improvement over state-of-the-art methods, particularly in challenging scenarios with large dynamic objects, while reducing computational requirements by 5x compared to previous dynamic scene representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05152v1">Splatography: Sparse multi-view dynamic Gaussian Splatting for filmmaking challenges</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Deformable Gaussian Splatting (GS) accomplishes photorealistic dynamic 3-D reconstruction from dense multi-view video (MVV) by learning to deform a canonical GS representation. However, in filmmaking, tight budgets can result in sparse camera configurations, which limits state-of-the-art (SotA) methods when capturing complex dynamic features. To address this issue, we introduce an approach that splits the canonical Gaussians and deformation field into foreground and background components using a sparse set of masks for frames at t=0. Each representation is separately trained on different loss functions during canonical pre-training. Then, during dynamic training, different parameters are modeled for each deformation field following common filmmaking practices. The foreground stage contains diverse dynamic features so changes in color, position and rotation are learned. While, the background containing film-crew and equipment, is typically dimmer and less dynamic so only changes in point position are learned. Experiments on 3-D and 2.5-D entertainment datasets show that our method produces SotA qualitative and quantitative results; up to 3 PSNR higher with half the model size on 3-D scenes. Unlike the SotA and without the need for dense mask supervision, our method also produces segmented dynamic reconstructions including transparent and dynamic textures. Code and video comparisons are available online: https://interims-git.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05109v1">Efficient representation of 3D spatial data for defense-related applications</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Geospatial sensor data is essential for modern defense and security, offering indispensable 3D information for situational awareness. This data, gathered from sources like lidar sensors and optical cameras, allows for the creation of detailed models of operational environments. In this paper, we provide a comparative analysis of traditional representation methods, such as point clouds, voxel grids, and triangle meshes, alongside modern neural and implicit techniques like Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS). Our evaluation reveals a fundamental trade-off: traditional models offer robust geometric accuracy ideal for functional tasks like line-of-sight analysis and physics simulations, while modern methods excel at producing high-fidelity, photorealistic visuals but often lack geometric reliability. Based on these findings, we conclude that a hybrid approach is the most promising path forward. We propose a system architecture that combines a traditional mesh scaffold for geometric integrity with a neural representation like 3DGS for visual detail, managed within a hierarchical scene structure to ensure scalability and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18090v2">GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted at NeurIPS 2025 (Spotlight). Project page: https://fictionarry.github.io/GeoSVR-project/
    </div>
    <details class="paper-abstract">
      Reconstructing accurate surfaces with radiance fields has achieved remarkable progress in recent years. However, prevailing approaches, primarily based on Gaussian Splatting, are increasingly constrained by representational bottlenecks. In this paper, we introduce GeoSVR, an explicit voxel-based framework that explores and extends the under-investigated potential of sparse voxels for achieving accurate, detailed, and complete surface reconstruction. As strengths, sparse voxels support preserving the coverage completeness and geometric clarity, while corresponding challenges also arise from absent scene constraints and locality in surface refinement. To ensure correct scene convergence, we first propose a Voxel-Uncertainty Depth Constraint that maximizes the effect of monocular depth cues while presenting a voxel-oriented uncertainty to avoid quality degradation, enabling effective and robust scene constraints yet preserving highly accurate geometries. Subsequently, Sparse Voxel Surface Regularization is designed to enhance geometric consistency for tiny voxels and facilitate the voxel-based formation of sharp and accurate surfaces. Extensive experiments demonstrate our superior performance compared to existing methods across diverse challenging scenarios, excelling in geometric accuracy, detail preservation, and reconstruction completeness while maintaining high efficiency. Code is available at https://github.com/Fictionarry/GeoSVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10473v3">ControlGS: Consistent Structural Compression Control for Deployment-Aware Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a highly deployable real-time method for novel view synthesis. In practice, it requires a universal, consistent control mechanism that adjusts the trade-off between rendering quality and model compression without scene-specific tuning, enabling automated deployment across different device performances and communication bandwidths. In this work, we present ControlGS, a control-oriented optimization framework that maps the trade-off between Gaussian count and rendering quality to a continuous, scene-agnostic, and highly responsive control axis. Extensive experiments across a wide range of scene scales and types (from small objects to large outdoor scenes) demonstrate that, by adjusting a globally unified control hyperparameter, ControlGS can flexibly generate models biased toward either structural compactness or high fidelity, regardless of the specific scene scale or complexity, while achieving markedly higher rendering quality with the same or fewer Gaussians compared to potential competing methods. Project page: https://zhang-fengdi.github.io/ControlGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18533v2">On Scaling Up 3D Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 ICLR 2025 Oral; Homepage: https://daohanlu.github.io/scaling-up-3dgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly popular for 3D reconstruction due to its superior visual quality and rendering speed. However, 3DGS training currently occurs on a single GPU, limiting its ability to handle high-resolution and large-scale 3D reconstruction tasks due to memory constraints. We introduce Grendel, a distributed system designed to partition 3DGS parameters and parallelize computation across multiple GPUs. As each Gaussian affects a small, dynamic subset of rendered pixels, Grendel employs sparse all-to-all communication to transfer the necessary Gaussians to pixel partitions and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training with multiple views. We explore various optimization hyperparameter scaling strategies and find that a simple sqrt(batch size) scaling rule is highly effective. Evaluations using large-scale, high-resolution scenes show that Grendel enhances rendering quality by scaling up 3DGS parameters across multiple GPUs. On the Rubble dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPUs, compared to a PSNR of 26.28 using 11.2 million Gaussians on a single GPU. Grendel is an open-source project available at: https://github.com/nyu-systems/Grendel-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04951v1">CLM: Removing the GPU Memory Barrier for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to appear in the 2026 ACM International Conference on Architectural Support for Programming Languages and Operating Systems
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is an increasingly popular novel view synthesis approach due to its fast rendering time, and high-quality output. However, scaling 3DGS to large (or intricate) scenes is challenging due to its large memory requirement, which exceed most GPU's memory capacity. In this paper, we describe CLM, a system that allows 3DGS to render large scenes using a single consumer-grade GPU, e.g., RTX4090. It does so by offloading Gaussians to CPU memory, and loading them into GPU memory only when necessary. To reduce performance and communication overheads, CLM uses a novel offloading strategy that exploits observations about 3DGS's memory access pattern for pipelining, and thus overlap GPU-to-CPU communication, GPU computation and CPU computation. Furthermore, we also exploit observation about the access pattern to reduce communication volume. Our evaluation shows that the resulting implementation can render a large scene that requires 100 million Gaussians on a single RTX4090 and achieve state-of-the-art reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21890v2">Diffusion Denoised Hyperspectral Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to 3DV 2026
    </div>
    <details class="paper-abstract">
      Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements of samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can be used to render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of hyperspectral scenes across the full spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of DD-HGS. The results demonstrate that DD-HGS achieves new state-of-the-art performance among previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21890v2">Diffusion Denoised Hyperspectral Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to 3DV 2026
    </div>
    <details class="paper-abstract">
      Hyperspectral imaging (HSI) has been widely used in agricultural applications for non-destructive estimation of plant nutrient composition and precise determination of nutritional elements of samples. Recently, 3D reconstruction methods have been used to create implicit neural representations of HSI scenes, which can help localize the target object's nutrient composition spatially and spectrally. Neural Radiance Field (NeRF) is a cutting-edge implicit representation that can be used to render hyperspectral channel compositions of each spatial location from any viewing direction. However, it faces limitations in training time and rendering speed. In this paper, we propose Diffusion-Denoised Hyperspectral Gaussian Splatting (DD-HGS), which enhances the state-of-the-art 3D Gaussian Splatting (3DGS) method with wavelength-aware spherical harmonics, a Kullback-Leibler divergence-based spectral loss, and a diffusion-based denoiser to enable 3D explicit reconstruction of hyperspectral scenes across the full spectral range. We present extensive evaluations on diverse real-world hyperspectral scenes from the Hyper-NeRF dataset to show the effectiveness of DD-HGS. The results demonstrate that DD-HGS achieves new state-of-the-art performance among previously published methods. Project page: https://dragonpg2000.github.io/DDHGS-website/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15680v2">Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://kywind.github.io/pgnd
    </div>
    <details class="paper-abstract">
      Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04665v1">Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Website: https://real2sim-eval.github.io/
    </div>
    <details class="paper-abstract">
      Robotic manipulation policies are advancing rapidly, but their direct evaluation in the real world remains costly, time-consuming, and difficult to reproduce, particularly for tasks involving deformable objects. Simulation provides a scalable and systematic alternative, yet existing simulators often fail to capture the coupled visual and physical complexity of soft-body interactions. We present a real-to-sim policy evaluation framework that constructs soft-body digital twins from real-world videos and renders robots, objects, and environments with photorealistic fidelity using 3D Gaussian Splatting. We validate our approach on representative deformable manipulation tasks, including plush toy packing, rope routing, and T-block pushing, demonstrating that simulated rollouts correlate strongly with real-world execution performance and reveal key behavioral patterns of learned policies. Our results suggest that combining physics-informed reconstruction with high-quality rendering enables reproducible, scalable, and accurate evaluation of robotic manipulation policies. Website: https://real2sim-eval.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16924v2">Optimized Minimal 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://maincold2.github.io/omg/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for real-time, high-performance rendering, enabling a wide range of applications. However, representing 3D scenes with numerous explicit Gaussian primitives imposes significant storage and memory overhead. Recent studies have shown that high-quality rendering can be achieved with a substantially reduced number of Gaussians when represented with high-precision attributes. Nevertheless, existing 3DGS compression methods still rely on a relatively large number of Gaussians, focusing primarily on attribute compression. This is because a smaller set of Gaussians becomes increasingly sensitive to lossy attribute compression, leading to severe quality degradation. Since the number of Gaussians is directly tied to computational costs, it is essential to reduce the number of Gaussians effectively rather than only optimizing storage. In this paper, we propose Optimized Minimal Gaussians representation (OMG), which significantly reduces storage while using a minimal number of primitives. First, we determine the distinct Gaussian from the near ones, minimizing redundancy without sacrificing quality. Second, we propose a compact and precise attribute representation that efficiently captures both continuity and irregularity among primitives. Additionally, we propose a sub-vector quantization technique for improved irregularity representation, maintaining fast training with a negligible codebook size. Extensive experiments demonstrate that OMG reduces storage requirements by nearly 50% compared to the previous state-of-the-art and enables 600+ FPS rendering while maintaining high rendering quality. Our source code is available at https://maincold2.github.io/omg/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13713v3">SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
    </div>
    <details class="paper-abstract">
      Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01110v3">A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks -- a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously. We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU -- without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to support real-time streaming and rendering. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes -- from broad aerial views to fine-grained ground-level details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04283v1">FastGS: Training 3D Gaussian Splatting in 100 Seconds</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Project page: https://fastgs.github.io/
    </div>
    <details class="paper-abstract">
      The dominant 3D Gaussian splatting (3DGS) acceleration methods fail to properly regulate the number of Gaussians during training, causing redundant computational time overhead. In this paper, we propose FastGS, a novel, simple, and general acceleration framework that fully considers the importance of each Gaussian based on multi-view consistency, efficiently solving the trade-off between training time and rendering quality. We innovatively design a densification and pruning strategy based on multi-view consistency, dispensing with the budgeting mechanism. Extensive experiments on Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets demonstrate that our method significantly outperforms the state-of-the-art methods in training speed, achieving a 3.32$\times$ training acceleration and comparable rendering quality compared with DashGaussian on the Mip-NeRF 360 dataset and a 15.45$\times$ acceleration compared with vanilla 3DGS on the Deep Blending dataset. We demonstrate that FastGS exhibits strong generality, delivering 2-7$\times$ training acceleration across various tasks, including dynamic scene reconstruction, surface reconstruction, sparse-view reconstruction, large-scale reconstruction, and simultaneous localization and mapping. The project page is available at https://fastgs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03992v1">CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01826v3">GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Synthesizing radio-frequency (RF) data given the transmitter and receiver positions, e.g., received signal strength indicator (RSSI), is critical for wireless networking and sensing applications, such as indoor localization. However, it remains challenging due to complex propagation interactions, including reflection, diffraction, and scattering. State-of-the-art neural radiance field (NeRF)-based methods achieve high-fidelity RF data synthesis but are limited by long training times and high inference latency. We introduce GSRF, a framework that extends 3D Gaussian Splatting (3DGS) from the optical domain to the RF domain, enabling efficient RF data synthesis. GSRF realizes this adaptation through three key innovations: First, it introduces complex-valued 3D Gaussians with a hybrid Fourier-Legendre basis to model directional and phase-dependent radiance. Second, it employs orthographic splatting for efficient ray-Gaussian intersection identification. Third, it incorporates a complex-valued ray tracing algorithm, executed on RF-customized CUDA kernels and grounded in wavefront propagation principles, to synthesize RF data in real time. Evaluated across various RF technologies, GSRF preserves high-fidelity RF data synthesis while achieving significant improvements in training efficiency, shorter training time, and reduced inference latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04797v1">3D Gaussian Point Encoders</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 10 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      In this work, we introduce the 3D Gaussian Point Encoder, an explicit per-point embedding built on mixtures of learned 3D Gaussians. This explicit geometric representation for 3D recognition tasks is a departure from widely used implicit representations such as PointNet. However, it is difficult to learn 3D Gaussian encoders in end-to-end fashion with standard optimizers. We develop optimization techniques based on natural gradients and distillation from PointNets to find a Gaussian Basis that can reconstruct PointNet activations. The resulting 3D Gaussian Point Encoders are faster and more parameter efficient than traditional PointNets. As in the 3D reconstruction literature where there has been considerable interest in the move from implicit (e.g., NeRF) to explicit (e.g., Gaussian Splatting) representations, we can take advantage of computational geometry heuristics to accelerate 3D Gaussian Point Encoders further. We extend filtering techniques from 3D Gaussian Splatting to construct encoders that run 2.7 times faster as a comparable accuracy PointNet while using 46% less memory and 88% fewer FLOPs. Furthermore, we demonstrate the effectiveness of 3D Gaussian Point Encoders as a component in Mamba3D, running 1.27 times faster and achieving a reduction in memory and FLOPs by 42% and 54% respectively. 3D Gaussian Point Encoders are lightweight enough to achieve high framerates on CPU-only devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23734v3">ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 NeurIPS 2025, Project Page: https://lhmd.top/zpressor, Code: https://github.com/ziplab/ZPressor
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) models have recently emerged as a promising solution for novel view synthesis, enabling one-pass inference without the need for per-scene 3DGS optimization. However, their scalability is fundamentally constrained by the limited capacity of their models, leading to degraded performance or excessive memory consumption as the number of input views increases. In this work, we analyze feed-forward 3DGS frameworks through the lens of the Information Bottleneck principle and introduce ZPressor, a lightweight architecture-agnostic module that enables efficient compression of multi-view inputs into a compact latent state $Z$ that retains essential scene information while discarding redundancy. Concretely, ZPressor enables existing feed-forward 3DGS models to scale to over 100 input views at 480P resolution on an 80GB GPU, by partitioning the views into anchor and support sets and using cross attention to compress the information from the support views into anchor views, forming the compressed latent state $Z$. We show that integrating ZPressor into several state-of-the-art feed-forward 3DGS models consistently improves performance under moderate input views and enhances robustness under dense view settings on two large-scale benchmarks DL3DV-10K and RealEstate10K. The video results, code and trained models are available on our project page: https://lhmd.top/zpressor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04789v3">Object-X: Learning to Reconstruct Multi-Modal 3D Object Representations</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Learning effective multi-modal 3D representations of objects is essential for numerous applications, such as augmented reality and robotics. Existing methods often rely on task-specific embeddings that are tailored either for semantic understanding or geometric reconstruction. As a result, these embeddings typically cannot be decoded into explicit geometry and simultaneously reused across tasks. In this paper, we propose Object-X, a versatile multi-modal object representation framework capable of encoding rich object embeddings (e.g. images, point cloud, text) and decoding them back into detailed geometric and visual reconstructions. Object-X operates by geometrically grounding the captured modalities in a 3D voxel grid and learning an unstructured embedding fusing the information from the voxels with the object attributes. The learned embedding enables 3D Gaussian Splatting-based object reconstruction, while also supporting a range of downstream tasks, including scene alignment, single-image 3D object reconstruction, and localization. Evaluations on two challenging real-world datasets demonstrate that Object-X produces high-fidelity novel-view synthesis comparable to standard 3D Gaussian Splatting, while significantly improving geometric accuracy. Moreover, Object-X achieves competitive performance with specialized methods in scene alignment and localization. Critically, our object-centric descriptors require 3-4 orders of magnitude less storage compared to traditional image- or point cloud-based approaches, establishing Object-X as a scalable and highly practical solution for multi-modal 3D scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03099v1">DentalSplat: Dental Occlusion Novel View Synthesis from Sparse Intra-Oral Photographs</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      In orthodontic treatment, particularly within telemedicine contexts, observing patients' dental occlusion from multiple viewpoints facilitates timely clinical decision-making. Recent advances in 3D Gaussian Splatting (3DGS) have shown strong potential in 3D reconstruction and novel view synthesis. However, conventional 3DGS pipelines typically rely on densely captured multi-view inputs and precisely initialized camera poses, limiting their practicality. Orthodontic cases, in contrast, often comprise only three sparse images, specifically, the anterior view and bilateral buccal views, rendering the reconstruction task especially challenging. The extreme sparsity of input views severely degrades reconstruction quality, while the absence of camera pose information further complicates the process. To overcome these limitations, we propose DentalSplat, an effective framework for 3D reconstruction from sparse orthodontic imagery. Our method leverages a prior-guided dense stereo reconstruction model to initialize the point cloud, followed by a scale-adaptive pruning strategy to improve the training efficiency and reconstruction quality of 3DGS. In scenarios with extremely sparse viewpoints, we further incorporate optical flow as a geometric constraint, coupled with gradient regularization, to enhance rendering fidelity. We validate our approach on a large-scale dataset comprising 950 clinical cases and an additional video-based test set of 195 cases designed to simulate real-world remote orthodontic imaging conditions. Experimental results demonstrate that our method effectively handles sparse input scenarios and achieves superior novel view synthesis quality for dental occlusion visualization, outperforming state-of-the-art techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11878v2">GS-Verse: Mesh-based Gaussian Splatting for Physics-aware Interaction in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      As the demand for immersive 3D content grows, the need for intuitive and efficient interaction methods becomes paramount. Current techniques for physically manipulating 3D content within Virtual Reality (VR) often face significant limitations, including reliance on engineering-intensive processes and simplified geometric representations, such as tetrahedral cages, which can compromise visual fidelity and physical accuracy. In this paper, we introduce GS-Verse (Gaussian Splatting for Virtual Environment Rendering and Scene Editing), a novel method designed to overcome these challenges by directly integrating an object's mesh with a Gaussian Splatting (GS) representation. Our approach enables more precise surface approximation, leading to highly realistic deformations and interactions. By leveraging existing 3D mesh assets, GS-Verse facilitates seamless content reuse and simplifies the development workflow. Moreover, our system is designed to be physics-engine-agnostic, granting developers robust deployment flexibility. This versatile architecture delivers a highly realistic, adaptable, and intuitive approach to interactive 3D manipulation. We rigorously validate our method against the current state-of-the-art technique that couples VR with GS in a comparative user study involving 18 participants. Specifically, we demonstrate that our approach is statistically significantly better for physics-aware stretching manipulation and is also more consistent in other physics-based manipulations like twisting and shaking. Further evaluation across various interactions and scenes confirms that our method consistently delivers high and reliable performance, showing its potential as a plausible alternative to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02777v1">PercHead: Perceptual Head Model for Single-Image 3D Head Reconstruction & Editing</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Project Page: https://antoniooroz.github.io/PercHead/ Video: https://www.youtube.com/watch?v=4hFybgTk4kE
    </div>
    <details class="paper-abstract">
      We present PercHead, a method for single-image 3D head reconstruction and semantic 3D editing - two tasks that are inherently challenging due to severe view occlusions, weak perceptual supervision, and the ambiguity of editing in 3D space. We develop a unified base model for reconstructing view-consistent 3D heads from a single input image. The model employs a dual-branch encoder followed by a ViT-based decoder that lifts 2D features into 3D space through iterative cross-attention. Rendering is performed using Gaussian Splatting. At the heart of our approach is a novel perceptual supervision strategy based on DINOv2 and SAM2.1, which provides rich, generalized signals for both geometric and appearance fidelity. Our model achieves state-of-the-art performance in novel-view synthesis and, furthermore, exhibits exceptional robustness to extreme viewing angles compared to established baselines. Furthermore, this base model can be seamlessly extended for semantic 3D editing by swapping the encoder and finetuning the network. In this variant, we disentangle geometry and style through two distinct input modalities: a segmentation map to control geometry and either a text prompt or a reference image to specify appearance. We highlight the intuitive and powerful 3D editing capabilities of our model through a lightweight, interactive GUI, where users can effortlessly sculpt geometry by drawing segmentation maps and stylize appearance via natural language or image prompts. Project Page: https://antoniooroz.github.io/PercHead Video: https://www.youtube.com/watch?v=4hFybgTk4kE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14501v4">Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
    </div>
    <details class="paper-abstract">
      3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02207v1">Object-Centric 3D Gaussian Splatting for Strawberry Plant Reconstruction and Phenotyping</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 11 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Strawberries are among the most economically significant fruits in the United States, generating over $2 billion in annual farm-gate sales and accounting for approximately 13% of the total fruit production value. Plant phenotyping plays a vital role in selecting superior cultivars by characterizing plant traits such as morphology, canopy structure, and growth dynamics. However, traditional plant phenotyping methods are time-consuming, labor-intensive, and often destructive. Recently, neural rendering techniques, notably Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have emerged as powerful frameworks for high-fidelity 3D reconstruction. By capturing a sequence of multi-view images or videos around a target plant, these methods enable non-destructive reconstruction of complex plant architectures. Despite their promise, most current applications of 3DGS in agricultural domains reconstruct the entire scene, including background elements, which introduces noise, increases computational costs, and complicates downstream trait analysis. To address this limitation, we propose a novel object-centric 3D reconstruction framework incorporating a preprocessing pipeline that leverages the Segment Anything Model v2 (SAM-2) and alpha channel background masking to achieve clean strawberry plant reconstructions. This approach produces more accurate geometric representations while substantially reducing computational time. With a background-free reconstruction, our algorithm can automatically estimate important plant traits, such as plant height and canopy width, using DBSCAN clustering and Principal Component Analysis (PCA). Experimental results show that our method outperforms conventional pipelines in both accuracy and efficiency, offering a scalable and non-destructive solution for strawberry plant phenotyping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01619v2">3DBonsai: Structure-Aware Bonsai Modeling Using Conditioned 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation have shown remarkable results by leveraging 3D priors in combination with 2D diffusion. However, previous methods utilize 3D priors that lack detailed and complex structural information, limiting them to generating simple objects and presenting challenges for creating intricate structures such as bonsai. In this paper, we propose 3DBonsai, a novel text-to-3D framework for generating 3D bonsai with complex structures. Technically, we first design a trainable 3D space colonization algorithm to produce bonsai structures, which are then enhanced through random sampling and point cloud augmentation to serve as the 3D Gaussian priors. We introduce two bonsai generation pipelines with distinct structural levels: fine structure conditioned generation, which initializes 3D Gaussians using a 3D structure prior to produce detailed and complex bonsai, and coarse structure conditioned generation, which employs a multi-view structure consistency module to align 2D and 3D structures. Moreover, we have compiled a unified 2D and 3D Chinese-style bonsai dataset. Our experimental results demonstrate that 3DBonsai significantly outperforms existing methods, providing a new benchmark for structure-aware 3D bonsai generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.11252v2">Anti-Aliased 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 NeurIPS 2025. Code will be available at https://github.com/maeyounes/AA-2DGS
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting (2DGS) has recently emerged as a promising method for novel view synthesis and surface reconstruction, offering better view-consistency and geometric accuracy than volumetric 3DGS. However, 2DGS suffers from severe aliasing artifacts when rendering at different sampling rates than those used during training, limiting its practical applications in scenarios requiring camera zoom or varying fields of view. We identify that these artifacts stem from two key limitations: the lack of frequency constraints in the representation and an ineffective screen-space clamping approach. To address these issues, we present AA-2DGS, an anti-aliased formulation of 2D Gaussian Splatting that maintains its geometric benefits while significantly enhancing rendering quality across different scales. Our method introduces a world-space flat smoothing kernel that constrains the frequency content of 2D Gaussian primitives based on the maximal sampling frequency from training views, effectively eliminating high-frequency artifacts when zooming in. Additionally, we derive a novel object-space Mip filter by leveraging an affine approximation of the ray-splat intersection mapping, which allows us to efficiently apply proper anti-aliasing directly in the local space of each splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00159v3">SonarSplat: Novel View Synthesis of Imaging Sonar via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      In this paper, we present SonarSplat, a novel Gaussian splatting framework for imaging sonar that demonstrates realistic novel view synthesis and models acoustic streaking phenomena. Our method represents the scene as a set of 3D Gaussians with acoustic reflectance and saturation properties. We develop a novel method to efficiently rasterize Gaussians to produce a range/azimuth image that is faithful to the acoustic image formation model of imaging sonar. In particular, we develop a novel approach to model azimuth streaking in a Gaussian splatting framework. We evaluate SonarSplat using real-world datasets of sonar images collected from an underwater robotic platform in a controlled test tank and in a real-world river environment. Compared to the state-of-the-art, SonarSplat offers improved image synthesis capabilities (+3.2 dB PSNR) and more accurate 3D reconstruction (77% lower Chamfer Distance). We also demonstrate that SonarSplat can be leveraged for azimuth streak removal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14270v2">GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Scene reconstruction has emerged as a central challenge in computer vision, with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting achieving remarkable progress. While Gaussian Splatting demonstrates strong performance on large-scale datasets, it often struggles to capture fine details or maintain realism in regions with sparse coverage, largely due to the inherent limitations of sparse 3D training data. In this work, we propose GauSSmart, a hybrid method that effectively bridges 2D foundational models and 3D Gaussian Splatting reconstruction. Our approach integrates established 2D computer vision techniques, including convex filtering and semantic feature supervision from foundational models such as DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D segmentation priors and high-dimensional feature embeddings, our method guides the densification and refinement of Gaussian splats, improving coverage in underrepresented areas and preserving intricate structural details. We validate our approach across three datasets, where GauSSmart consistently outperforms existing Gaussian Splatting in the majority of evaluated scenes. Our results demonstrate the significant potential of hybrid 2D-3D approaches, highlighting how the thoughtful combination of 2D foundational models with 3D reconstruction pipelines can overcome the limitations inherent in either approach alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19588v2">Gaussian Splashing: Direct Volumetric Rendering Underwater</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      In underwater images, most useful features are occluded by water. The extent of the occlusion depends on imaging geometry and can vary even across a sequence of burst images. As a result, 3D reconstruction methods robust on in-air scenes, like Neural Radiance Field methods (NeRFs) or 3D Gaussian Splatting (3DGS), fail on underwater scenes. While a recent underwater adaptation of NeRFs achieved state-of-the-art results, it is impractically slow: reconstruction takes hours and its rendering rate, in frames per second (FPS), is less than 1. Here, we present a new method that takes only a few minutes for reconstruction and renders novel underwater scenes at 140 FPS. Named Gaussian Splashing, our method unifies the strengths and speed of 3DGS with an image formation model for capturing scattering, introducing innovations in the rendering and depth estimation procedures and in the 3DGS loss function. Despite the complexities of underwater adaptation, our method produces images at unparalleled speeds with superior details. Moreover, it reveals distant scene details with far greater clarity than other methods, dramatically improving reconstructed and rendered images. We demonstrate results on existing datasets and a new dataset we have collected. Additional visual results are available at: https://bgu-cs-vil.github.io/gaussiansplashingUW.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25173v2">D$^2$GS: Dense Depth Regularization for LiDAR-free Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Recently, Gaussian Splatting (GS) has shown great potential for urban scene reconstruction in the field of autonomous driving. However, current urban scene reconstruction methods often depend on multimodal sensors as inputs, \textit{i.e.} LiDAR and images. Though the geometry prior provided by LiDAR point clouds can largely mitigate ill-posedness in reconstruction, acquiring such accurate LiDAR data is still challenging in practice: i) precise spatiotemporal calibration between LiDAR and other sensors is required, as they may not capture data simultaneously; ii) reprojection errors arise from spatial misalignment when LiDAR and cameras are mounted at different locations. To avoid the difficulty of acquiring accurate LiDAR depth, we propose D$^2$GS, a LiDAR-free urban scene reconstruction framework. In this work, we obtain geometry priors that are as effective as LiDAR while being denser and more accurate. $\textbf{First}$, we initialize a dense point cloud by back-projecting multi-view metric depth predictions. This point cloud is then optimized by a Progressive Pruning strategy to improve the global consistency. $\textbf{Second}$, we jointly refine Gaussian geometry and predicted dense metric depth via a Depth Enhancer. Specifically, we leverage diffusion priors from a depth foundation model to enhance the depth maps rendered by Gaussians. In turn, the enhanced depths provide stronger geometric constraints during Gaussian training. $\textbf{Finally}$, we improve the accuracy of ground geometry by constraining the shape and normal attributes of Gaussians within road regions. Extensive experiments on the Waymo dataset demonstrate that our method consistently outperforms state-of-the-art methods, producing more accurate geometry even when compared with those using ground-truth LiDAR data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11252v2">Anti-Aliased 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 NeurIPS 2025. Code will be available at https://github.com/maeyounes/AA-2DGS
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting (2DGS) has recently emerged as a promising method for novel view synthesis and surface reconstruction, offering better view-consistency and geometric accuracy than volumetric 3DGS. However, 2DGS suffers from severe aliasing artifacts when rendering at different sampling rates than those used during training, limiting its practical applications in scenarios requiring camera zoom or varying fields of view. We identify that these artifacts stem from two key limitations: the lack of frequency constraints in the representation and an ineffective screen-space clamping approach. To address these issues, we present AA-2DGS, an anti-aliased formulation of 2D Gaussian Splatting that maintains its geometric benefits while significantly enhancing rendering quality across different scales. Our method introduces a world-space flat smoothing kernel that constrains the frequency content of 2D Gaussian primitives based on the maximal sampling frequency from training views, effectively eliminating high-frequency artifacts when zooming in. Additionally, we derive a novel object-space Mip filter by leveraging an affine approximation of the ray-splat intersection mapping, which allows us to efficiently apply proper anti-aliasing directly in the local space of each splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00560v1">4D Neural Voxel Splatting: Dynamic Scene Rendering with Voxelized Guassian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3D-GS) achieves efficient rendering for novel view synthesis, extending it to dynamic scenes still results in substantial memory overhead from replicating Gaussians across frames. To address this challenge, we propose 4D Neural Voxel Splatting (4D-NVS), which combines voxel-based representations with neural Gaussian splatting for efficient dynamic scene modeling. Instead of generating separate Gaussian sets per timestamp, our method employs a compact set of neural voxels with learned deformation fields to model temporal dynamics. The design greatly reduces memory consumption and accelerates training while preserving high image quality. We further introduce a novel view refinement stage that selectively improves challenging viewpoints through targeted optimization, maintaining global efficiency while enhancing rendering quality for difficult viewing angles. Experiments demonstrate that our method outperforms state-of-the-art approaches with significant memory reduction and faster training, enabling real-time rendering with superior visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11252v2">Anti-Aliased 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-11-01
      | 💬 NeurIPS 2025. Code will be available at https://github.com/maeyounes/AA-2DGS
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting (2DGS) has recently emerged as a promising method for novel view synthesis and surface reconstruction, offering better view-consistency and geometric accuracy than volumetric 3DGS. However, 2DGS suffers from severe aliasing artifacts when rendering at different sampling rates than those used during training, limiting its practical applications in scenarios requiring camera zoom or varying fields of view. We identify that these artifacts stem from two key limitations: the lack of frequency constraints in the representation and an ineffective screen-space clamping approach. To address these issues, we present AA-2DGS, an anti-aliased formulation of 2D Gaussian Splatting that maintains its geometric benefits while significantly enhancing rendering quality across different scales. Our method introduces a world-space flat smoothing kernel that constrains the frequency content of 2D Gaussian primitives based on the maximal sampling frequency from training views, effectively eliminating high-frequency artifacts when zooming in. Additionally, we derive a novel object-space Mip filter by leveraging an affine approximation of the ray-splat intersection mapping, which allows us to efficiently apply proper anti-aliasing directly in the local space of each splat.
    </details>
</div>
