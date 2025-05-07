# gaussian splatting - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00814v3">VR-Doh: Hands-on 3D Modeling in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      We introduce VR-Doh, an open-source, hands-on 3D modeling system that enables intuitive creation and manipulation of elastoplastic objects in Virtual Reality (VR). By customizing the Material Point Method (MPM) for real-time simulation of hand-induced large deformations and enhancing 3D Gaussian Splatting for seamless rendering, VR-Doh provides an interactive and immersive 3D modeling experience. Users can naturally sculpt, deform, and edit objects through both contact- and gesture-based hand-object interactions. To achieve real-time performance, our system incorporates localized simulation techniques, particle-level collision handling, and the decoupling of physical and appearance representations, ensuring smooth and responsive interactions. VR-Doh supports both object creation and editing, enabling diverse modeling tasks such as designing food items, characters, and interlocking structures, all resulting in simulation-ready assets. User studies with both novice and experienced participants highlight the system's intuitive design, immersive feedback, and creative potential. Compared to existing geometric modeling tools, VR-Doh offers enhanced accessibility and natural interaction, making it a powerful tool for creative exploration in VR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03310v1">3D Gaussian Splatting Data Compression with Mixture of Priors</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) data compression is crucial for enabling efficient storage and transmission in 3D scene modeling. However, its development remains limited due to inadequate entropy models and suboptimal quantization strategies for both lossless and lossy compression scenarios, where existing methods have yet to 1) fully leverage hyperprior information to construct robust conditional entropy models, and 2) apply fine-grained, element-wise quantization strategies for improved compression granularity. In this work, we propose a novel Mixture of Priors (MoP) strategy to simultaneously address these two challenges. Specifically, inspired by the Mixture-of-Experts (MoE) paradigm, our MoP approach processes hyperprior information through multiple lightweight MLPs to generate diverse prior features, which are subsequently integrated into the MoP feature via a gating mechanism. To enhance lossless compression, the resulting MoP feature is utilized as a hyperprior to improve conditional entropy modeling. Meanwhile, for lossy compression, we employ the MoP feature as guidance information in an element-wise quantization procedure, leveraging a prior-guided Coarse-to-Fine Quantization (C2FQ) strategy with a predefined quantization step value. Specifically, we expand the quantization step value into a matrix and adaptively refine it from coarse to fine granularity, guided by the MoP feature, thereby obtaining a quantization step matrix that facilitates element-wise quantization. Extensive experiments demonstrate that our proposed 3DGS data compression framework achieves state-of-the-art performance across multiple benchmarks, including Mip-NeRF360, BungeeNeRF, DeepBlending, and Tank&Temples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18630v2">Deformable Beta Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 SIGGRAPH 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has advanced radiance field reconstruction by enabling real-time rendering. However, its reliance on Gaussian kernels for geometry and low-order Spherical Harmonics (SH) for color encoding limits its ability to capture complex geometries and diverse colors. We introduce Deformable Beta Splatting (DBS), a deformable and compact approach that enhances both geometry and color representation. DBS replaces Gaussian kernels with deformable Beta Kernels, which offer bounded support and adaptive frequency control to capture fine geometric details with higher fidelity while achieving better memory efficiency. In addition, we extended the Beta Kernel to color encoding, which facilitates improved representation of diffuse and specular components, yielding superior results compared to SH-based methods. Furthermore, Unlike prior densification techniques that depend on Gaussian properties, we mathematically prove that adjusting regularized opacity alone ensures distribution-preserved Markov chain Monte Carlo (MCMC), independent of the splatting kernel type. Experimental results demonstrate that DBS achieves state-of-the-art visual quality while utilizing only 45% of the parameters and rendering 1.5x faster than 3DGS-MCMC, highlighting the superior performance of DBS for real-time radiance field rendering. Interactive demonstrations and source code are available on our project website: https://rongliu-leo.github.io/beta-splatting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07577v2">3D Vision-Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted at ICLR 2025. Main paper + supplementary material
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D reconstruction methods and vision-language models have propelled the development of multi-modal 3D scene understanding, which has vital applications in robotics, autonomous driving, and virtual/augmented reality. However, current multi-modal scene understanding approaches have naively embedded semantic representations into 3D reconstruction methods without striking a balance between visual and language modalities, which leads to unsatisfying semantic rasterization of translucent or reflective objects, as well as over-fitting on color modality. To alleviate these limitations, we propose a solution that adequately handles the distinct visual and semantic modalities, i.e., a 3D vision-language Gaussian splatting model for scene understanding, to put emphasis on the representation learning of language modality. We propose a novel cross-modal rasterizer, using modality fusion along with a smoothed semantic indicator for enhancing semantic rasterization. We also employ a camera-view blending technique to improve semantic consistency between existing and synthesized views, thereby effectively mitigating over-fitting. Extensive experiments demonstrate that our method achieves state-of-the-art performance in open-vocabulary semantic segmentation, surpassing existing methods by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18468v3">RGS-DR: Reflective Gaussian Surfels with Deferred Rendering for Shiny Objects</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      We introduce RGS-DR, a novel inverse rendering method for reconstructing and rendering glossy and reflective objects with support for flexible relighting and scene editing. Unlike existing methods (e.g., NeRF and 3D Gaussian Splatting), which struggle with view-dependent effects, RGS-DR utilizes a 2D Gaussian surfel representation to accurately estimate geometry and surface normals, an essential property for high-quality inverse rendering. Our approach explicitly models geometric and material properties through learnable primitives rasterized into a deferred shading pipeline, effectively reducing rendering artifacts and preserving sharp reflections. By employing a multi-level cube mipmap, RGS-DR accurately approximates environment lighting integrals, facilitating high-quality reconstruction and relighting. A residual pass with spherical-mipmap-based directional encoding further refines the appearance modeling. Experiments demonstrate that RGS-DR achieves high-quality reconstruction and rendering quality for shiny objects, often outperforming reconstruction-exclusive state-of-the-art methods incapable of relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00159v2">SonarSplat: Novel View Synthesis of Imaging Sonar via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      In this paper, we present SonarSplat, a novel Gaussian splatting framework for imaging sonar that demonstrates realistic novel view synthesis and models acoustic streaking phenomena. Our method represents the scene as a set of 3D Gaussians with acoustic reflectance and saturation properties. We develop a novel method to efficiently rasterize Gaussians to produce a range/azimuth image that is faithful to the acoustic image formation model of imaging sonar. In particular, we develop a novel approach to model azimuth streaking in a Gaussian splatting framework. We evaluate SonarSplat using real-world datasets of sonar images collected from an underwater robotic platform in a controlled test tank and in a real-world river environment. Compared to the state-of-the-art, SonarSplat offers improved image synthesis capabilities (+3.2 dB PSNR) and more accurate 3D reconstruction (52% lower Chamfer Distance). We also demonstrate that SonarSplat can be leveraged for azimuth streak removal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02178v1">Sparfels: Fast Reconstruction from Sparse Unposed Imagery</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Project page : https://shubhendu-jena.github.io/Sparfels/
    </div>
    <details class="paper-abstract">
      We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02175v1">SparSplat: Fast Multi-View Reconstruction with Generalizable 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 Project page : https://shubhendu-jena.github.io/SparSplat/
    </div>
    <details class="paper-abstract">
      Recovering 3D information from scenes via multi-view stereo reconstruction (MVS) and novel view synthesis (NVS) is inherently challenging, particularly in scenarios involving sparse-view setups. The advent of 3D Gaussian Splatting (3DGS) enabled real-time, photorealistic NVS. Following this, 2D Gaussian Splatting (2DGS) leveraged perspective accurate 2D Gaussian primitive rasterization to achieve accurate geometry representation during rendering, improving 3D scene reconstruction while maintaining real-time performance. Recent approaches have tackled the problem of sparse real-time NVS using 3DGS within a generalizable, MVS-based learning framework to regress 3D Gaussian parameters. Our work extends this line of research by addressing the challenge of generalizable sparse 3D reconstruction and NVS jointly, and manages to perform successfully at both tasks. We propose an MVS-based learning pipeline that regresses 2DGS surface element parameters in a feed-forward fashion to perform 3D shape reconstruction and NVS from sparse-view images. We further show that our generalizable pipeline can benefit from preexisting foundational multi-view deep visual features. The resulting model attains the state-of-the-art results on the DTU sparse 3D reconstruction benchmark in terms of Chamfer distance to ground-truth, as-well as state-of-the-art NVS. It also demonstrates strong generalization on the BlendedMVS and Tanks and Temples datasets. We note that our model outperforms the prior state-of-the-art in feed-forward sparse view reconstruction based on volume rendering of implicit representations, while offering an almost 2 orders of magnitude higher inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02126v1">GarmentGS: Point-Cloud Guided Gaussian Splatting for High-Fidelity Non-Watertight 3D Garment Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      Traditional 3D garment creation requires extensive manual operations, resulting in time and labor costs. Recently, 3D Gaussian Splatting has achieved breakthrough progress in 3D scene reconstruction and rendering, attracting widespread attention and opening new pathways for 3D garment reconstruction. However, due to the unstructured and irregular nature of Gaussian primitives, it is difficult to reconstruct high-fidelity, non-watertight 3D garments. In this paper, we present GarmentGS, a dense point cloud-guided method that can reconstruct high-fidelity garment surfaces with high geometric accuracy and generate non-watertight, single-layer meshes. Our method introduces a fast dense point cloud reconstruction module that can complete garment point cloud reconstruction in 10 minutes, compared to traditional methods that require several hours. Furthermore, we use dense point clouds to guide the movement, flattening, and rotation of Gaussian primitives, enabling better distribution on the garment surface to achieve superior rendering effects and geometric accuracy. Through numerical and visual comparisons, our method achieves fast training and real-time rendering while maintaining competitive quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02108v1">SignSplat: Rendering Sign Language via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
    </div>
    <details class="paper-abstract">
      State-of-the-art approaches for conditional human body rendering via Gaussian splatting typically focus on simple body motions captured from many views. This is often in the context of dancing or walking. However, for more complex use cases, such as sign language, we care less about large body motion and more about subtle and complex motions of the hands and face. The problems of building high fidelity models are compounded by the complexity of capturing multi-view data of sign. The solution is to make better use of sequence data, ensuring that we can overcome the limited information from only a few views by exploiting temporal variability. Nevertheless, learning from sequence-level data requires extremely accurate and consistent model fitting to ensure that appearance is consistent across complex motions. We focus on how to achieve this, constraining mesh parameters to build an accurate Gaussian splatting framework from few views capable of modelling subtle human motion. We leverage regularization techniques on the Gaussian parameters to mitigate overfitting and rendering artifacts. Additionally, we propose a new adaptive control method to densify Gaussians and prune splat points on the mesh surface. To demonstrate the accuracy of our approach, we render novel sequences of sign language video, building on neural machine translation approaches to sign stitching. On benchmark datasets, our approach achieves state-of-the-art performance; and on highly articulated and complex sign language motion, we significantly outperform competing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02720v4">3D-HGS: 3D Half-Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-04
      | 💬 8 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Photo-realistic image rendering from 3D scene reconstruction has advanced significantly with neural rendering techniques. Among these, 3D Gaussian Splatting (3D-GS) outperforms Neural Radiance Fields (NeRFs) in quality and speed but struggles with shape and color discontinuities. We propose 3D Half-Gaussian (3D-HGS) kernels as a plug-and-play solution to address these limitations. Our experiments show that 3D-HGS enhances existing 3D-GS methods, achieving state-of-the-art rendering quality without compromising speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01938v1">HybridGS: High-Efficiency Gaussian Splatting Data Compression using Dual-Channel Sparse Representation and Point Cloud Encoder</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 Accepted by ICML2025
    </div>
    <details class="paper-abstract">
      Most existing 3D Gaussian Splatting (3DGS) compression schemes focus on producing compact 3DGS representation via implicit data embedding. They have long coding times and highly customized data format, making it difficult for widespread deployment. This paper presents a new 3DGS compression framework called HybridGS, which takes advantage of both compact generation and standardized point cloud data encoding. HybridGS first generates compact and explicit 3DGS data. A dual-channel sparse representation is introduced to supervise the primitive position and feature bit depth. It then utilizes a canonical point cloud encoder to perform further data compression and form standard output bitstreams. A simple and effective rate control scheme is proposed to pivot the interpretable data compression scheme. At the current stage, HybridGS does not include any modules aimed at improving 3DGS quality during generation. But experiment results show that it still provides comparable reconstruction performance against state-of-the-art methods, with evidently higher encoding and decoding speed. The code is publicly available at https://github.com/Qi-Yangsjtu/HybridGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01928v1">GenSync: A Generalized Talking Head Framework for Audio-driven Multi-Subject Lip-Sync using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      We introduce GenSync, a novel framework for multi-identity lip-synced video synthesis using 3D Gaussian Splatting. Unlike most existing 3D methods that require training a new model for each identity , GenSync learns a unified network that synthesizes lip-synced videos for multiple speakers. By incorporating a Disentanglement Module, our approach separates identity-specific features from audio representations, enabling efficient multi-identity video synthesis. This design reduces computational overhead and achieves 6.8x faster training compared to state-of-the-art models, while maintaining high lip-sync accuracy and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01869v1">Visual enhancement and 3D representation for underwater scenes: a review</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Underwater visual enhancement (UVE) and underwater 3D reconstruction pose significant challenges in computer vision and AI-based tasks due to complex imaging conditions in aquatic environments. Despite the development of numerous enhancement algorithms, a comprehensive and systematic review covering both UVE and underwater 3D reconstruction remains absent. To advance research in these areas, we present an in-depth review from multiple perspectives. First, we introduce the fundamental physical models, highlighting the peculiarities that challenge conventional techniques. We survey advanced methods for visual enhancement and 3D reconstruction specifically designed for underwater scenarios. The paper assesses various approaches from non-learning methods to advanced data-driven techniques, including Neural Radiance Fields and 3D Gaussian Splatting, discussing their effectiveness in handling underwater distortions. Finally, we conduct both quantitative and qualitative evaluations of state-of-the-art UVE and underwater 3D reconstruction algorithms across multiple benchmark datasets. Finally, we highlight key research directions for future advancements in underwater vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16693v2">PIN-WM: Learning Physics-INformed World Models for Non-Prehensile Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-05-03
      | 💬 Robotics: Science and Systems 2025
    </div>
    <details class="paper-abstract">
      While non-prehensile manipulation (e.g., controlled pushing/poking) constitutes a foundational robotic skill, its learning remains challenging due to the high sensitivity to complex physical interactions involving friction and restitution. To achieve robust policy learning and generalization, we opt to learn a world model of the 3D rigid body dynamics involved in non-prehensile manipulations and use it for model-based reinforcement learning. We propose PIN-WM, a Physics-INformed World Model that enables efficient end-to-end identification of a 3D rigid body dynamical system from visual observations. Adopting differentiable physics simulation, PIN-WM can be learned with only few-shot and task-agnostic physical interaction trajectories. Further, PIN-WM is learned with observational loss induced by Gaussian Splatting without needing state estimation. To bridge Sim2Real gaps, we turn the learned PIN-WM into a group of Digital Cousins via physics-aware randomizations which perturb physics and rendering parameters to generate diverse and meaningful variations of the PIN-WM. Extensive evaluations on both simulation and real-world tests demonstrate that PIN-WM, enhanced with physics-aware digital cousins, facilitates learning robust non-prehensile manipulation skills with Sim2Real transfer, surpassing the Real2Sim2Real state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01799v1">AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-03
    </div>
    <details class="paper-abstract">
      Underwater scene reconstruction is a critical tech-nology for underwater operations, enabling the generation of 3D models from images captured by underwater platforms. However, the quality of underwater images is often degraded due to medium interference, which limits the effectiveness of Structure-from-Motion (SfM) pose estimation, leading to subsequent reconstruction failures. Additionally, SfM methods typically operate at slower speeds, further hindering their applicability in real-time scenarios. In this paper, we introduce AquaGS, an SfM-free underwater scene reconstruction model based on the SeaThru algorithm, which facilitates rapid and accurate separation of scene details and medium features. Our approach initializes Gaussians by integrating state-of-the-art multi-view stereo (MVS) technology, employs implicit Neural Radiance Fields (NeRF) for rendering translucent media and utilizes the latest explicit 3D Gaussian Splatting (3DGS) technique to render object surfaces, which effectively addresses the limitations of traditional methods and accurately simulates underwater optical phenomena. Experimental results on the data set and the robot platform show that our model can complete high-precision reconstruction in 30 seconds with only 3 image inputs, significantly enhancing the practical application of the algorithm in robotic platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01383v1">FalconWing: An Open-Source Platform for Ultra-Light Fixed-Wing Aircraft Research</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      We present FalconWing -- an open-source, ultra-lightweight (150 g) fixed-wing platform for autonomy research. The hardware platform integrates a small camera, a standard airframe, offboard computation, and radio communication for manual overrides. We demonstrate FalconWing's capabilities by developing and deploying a purely vision-based control policy for autonomous landing (without IMU or motion capture) using a novel real-to-sim-to-real learning approach. Our learning approach: (1) constructs a photorealistic simulation environment via 3D Gaussian splatting trained on real-world images; (2) identifies nonlinear dynamics from vision-estimated real-flight data; and (3) trains a multi-modal Vision Transformer (ViT) policy through simulation-only imitation learning. The ViT architecture fuses single RGB image with the history of control actions via self-attention, preserving temporal context while maintaining real-time 20 Hz inference. When deployed zero-shot on the hardware platform, this policy achieves an 80% success rate in vision-based autonomous landings. Together with the hardware specifications, we also open-source the system dynamics, the software for photorealistic simulator and the learning approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01235v1">Compensating Spatiotemporally Inconsistent Observations for Online Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 SIGGRAPH 2025, Project page: https://bbangsik13.github.io/OR2
    </div>
    <details class="paper-abstract">
      Online reconstruction of dynamic scenes is significant as it enables learning scenes from live-streaming video inputs, while existing offline dynamic reconstruction methods rely on recorded video inputs. However, previous online reconstruction approaches have primarily focused on efficiency and rendering quality, overlooking the temporal consistency of their results, which often contain noticeable artifacts in static regions. This paper identifies that errors such as noise in real-world recordings affect temporal inconsistency in online reconstruction. We propose a method that enhances temporal consistency in online reconstruction from observations with temporal inconsistency which is inevitable in cameras. We show that our method restores the ideal observation by subtracting the learned error. We demonstrate that applying our method to various baselines significantly enhances both temporal consistency and rendering quality across datasets. Code, video results, and checkpoints are available at https://bbangsik13.github.io/OR2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00362v2">UDGS-SLAM : UniDepth Assisted Gaussian Splatting for Monocular SLAM</a></div>
    <div class="paper-meta">
      📅 2025-05-02
    </div>
    <details class="paper-abstract">
      Recent advancements in monocular neural depth estimation, particularly those achieved by the UniDepth network, have prompted the investigation of integrating UniDepth within a Gaussian splatting framework for monocular SLAM. This study presents UDGS-SLAM, a novel approach that eliminates the necessity of RGB-D sensors for depth estimation within Gaussian splatting framework. UDGS-SLAM employs statistical filtering to ensure local consistency of the estimated depth and jointly optimizes camera trajectory and Gaussian scene representation parameters. The proposed method achieves high-fidelity rendered images and low ATERMSE of the camera trajectory. The performance of UDGS-SLAM is rigorously evaluated using the TUM RGB-D dataset and benchmarked against several baseline methods, demonstrating superior performance across various scenarios. Additionally, an ablation study is conducted to validate design choices and investigate the impact of different network backbone encoders on system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15122v3">MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 The first two authors contributed equally to this work (equal contribution). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/mobgs-site/
    </div>
    <details class="paper-abstract">
      We present MoBGS, a novel deblurring dynamic 3D Gaussian Splatting (3DGS) framework capable of reconstructing sharp and high-quality novel spatio-temporal views from blurry monocular videos in an end-to-end manner. Existing dynamic novel view synthesis (NVS) methods are highly sensitive to motion blur in casually captured videos, resulting in significant degradation of rendering quality. While recent approaches address motion-blurred inputs for NVS, they primarily focus on static scene reconstruction and lack dedicated motion modeling for dynamic objects. To overcome these limitations, our MoBGS introduces a novel Blur-adaptive Latent Camera Estimation (BLCE) method for effective latent camera trajectory estimation, improving global camera motion deblurring. In addition, we propose a physically-inspired Latent Camera-induced Exposure Estimation (LCEE) method to ensure consistent deblurring of both global camera and local object motion. Our MoBGS framework ensures the temporal consistency of unseen latent timestamps and robust motion decomposition of static and dynamic regions. Extensive experiments on the Stereo Blur dataset and real-world blurry videos show that our MoBGS significantly outperforms the very recent advanced methods (DyBluRF and Deblur4DGS), achieving state-of-the-art performance for dynamic NVS under motion blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00421v1">Real-Time Animatable 2DGS-Avatars with Detail Enhancement from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      High-quality, animatable 3D human avatar reconstruction from monocular videos offers significant potential for reducing reliance on complex hardware, making it highly practical for applications in game development, augmented reality, and social media. However, existing methods still face substantial challenges in capturing fine geometric details and maintaining animation stability, particularly under dynamic or complex poses. To address these issues, we propose a novel real-time framework for animatable human avatar reconstruction based on 2D Gaussian Splatting (2DGS). By leveraging 2DGS and global SMPL pose parameters, our framework not only aligns positional and rotational discrepancies but also enables robust and natural pose-driven animation of the reconstructed avatars. Furthermore, we introduce a Rotation Compensation Network (RCN) that learns rotation residuals by integrating local geometric features with global pose parameters. This network significantly improves the handling of non-rigid deformations and ensures smooth, artifact-free pose transitions during animation. Experimental results demonstrate that our method successfully reconstructs realistic and highly animatable human avatars from monocular videos, effectively preserving fine-grained details while ensuring stable and natural pose variation. Our approach surpasses current state-of-the-art methods in both reconstruction quality and animation robustness on public benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18768v2">TransparentGS: Fast Inverse Rendering of Transparent Objects with Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 accepted by SIGGRAPH 2025; https://letianhuang.github.io/transparentgs/
    </div>
    <details class="paper-abstract">
      The emergence of neural and Gaussian-based radiance field methods has led to considerable advancements in novel view synthesis and 3D object reconstruction. Nonetheless, specular reflection and refraction continue to pose significant challenges due to the instability and incorrect overfitting of radiance fields to high-frequency light variations. Currently, even 3D Gaussian Splatting (3D-GS), as a powerful and efficient tool, falls short in recovering transparent objects with nearby contents due to the existence of apparent secondary ray effects. To address this issue, we propose TransparentGS, a fast inverse rendering pipeline for transparent objects based on 3D-GS. The main contributions are three-fold. Firstly, an efficient representation of transparent objects, transparent Gaussian primitives, is designed to enable specular refraction through a deferred refraction strategy. Secondly, we leverage Gaussian light field probes (GaussProbe) to encode both ambient light and nearby contents in a unified framework. Thirdly, a depth-based iterative probes query (IterQuery) algorithm is proposed to reduce the parallax errors in our probe-based framework. Experiments demonstrate the speed and accuracy of our approach in recovering transparent objects from complex environments, as well as several applications in computer graphics and vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20379v2">GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      In this paper, we present a method for localizing a query image with respect to a precomputed 3D Gaussian Splatting (3DGS) scene representation. First, the method uses 3DGS to render a synthetic RGBD image at some initial pose estimate. Second, it establishes 2D-2D correspondences between the query image and this synthetic image. Third, it uses the depth map to lift the 2D-2D correspondences to 2D-3D correspondences and solves a perspective-n-point (PnP) problem to produce a final pose estimate. Results from evaluation across three existing datasets with 38 scenes and over 2,700 test images show that our method significantly reduces both inference time (by over two orders of magnitude, from more than 10 seconds to as fast as 0.1 seconds) and estimation error compared to baseline methods that use photometric loss minimization. Results also show that our method tolerates large errors in the initial pose estimate of up to 55{\deg} in rotation and 1.1 units in translation (normalized by scene scale), achieving final pose errors of less than 5{\deg} in rotation and 0.05 units in translation on 90% of images from the Synthetic NeRF and Mip-NeRF360 datasets and on 42% of images from the more challenging Tanks and Temples dataset.
    </details>
</div>
