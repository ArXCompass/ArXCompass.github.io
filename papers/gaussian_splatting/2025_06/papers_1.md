# gaussian splatting - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22280v1">DIGS: Dynamic CBCT Reconstruction using Deformation-Informed 4D Gaussian Splatting and a Low-Rank Free-Form Deformation Model</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted by MICCAI 2025
    </div>
    <details class="paper-abstract">
      3D Cone-Beam CT (CBCT) is widely used in radiotherapy but suffers from motion artifacts due to breathing. A common clinical approach mitigates this by sorting projections into respiratory phases and reconstructing images per phase, but this does not account for breathing variability. Dynamic CBCT instead reconstructs images at each projection, capturing continuous motion without phase sorting. Recent advancements in 4D Gaussian Splatting (4DGS) offer powerful tools for modeling dynamic scenes, yet their application to dynamic CBCT remains underexplored. Existing 4DGS methods, such as HexPlane, use implicit motion representations, which are computationally expensive. While explicit low-rank motion models have been proposed, they lack spatial regularization, leading to inconsistencies in Gaussian motion. To address these limitations, we introduce a free-form deformation (FFD)-based spatial basis function and a deformation-informed framework that enforces consistency by coupling the temporal evolution of Gaussian's mean position, scale, and rotation under a unified deformation field. We evaluate our approach on six CBCT datasets, demonstrating superior image quality with a 6x speedup over HexPlane. These results highlight the potential of deformation-informed 4DGS for efficient, motion-compensated CBCT reconstruction. The code is available at https://github.com/Yuliang-Huang/DIGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22099v1">BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted at ICCV 2025, Project Page: https://github.com/fudan-zvg/BezierGS
    </div>
    <details class="paper-abstract">
      The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose B\'ezier curve Gaussian splatting (B\'ezierGS), which represents the motion trajectories of dynamic objects using learnable B\'ezier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that B\'ezierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22605v2">Audio-Plane: Audio Factorization Plane Gaussian Splatting for Real-Time Talking Head Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Demo video at \url{https://sstzal.github.io/Audio-Plane/}
    </div>
    <details class="paper-abstract">
      Talking head synthesis has emerged as a prominent research topic in computer graphics and multimedia, yet most existing methods often struggle to strike a balance between generation quality and computational efficiency, particularly under real-time constraints. In this paper, we propose a novel framework that integrates Gaussian Splatting with a structured Audio Factorization Plane (Audio-Plane) to enable high-quality, audio-synchronized, and real-time talking head generation. For modeling a dynamic talking head, a 4D volume representation, which consists of three axes in 3D space and one temporal axis aligned with audio progression, is typically required. However, directly storing and processing a dense 4D grid is impractical due to the high memory and computation cost, and lack of scalability for longer durations. We address this challenge by decomposing the 4D volume representation into a set of audio-independent spatial planes and audio-dependent planes, forming a compact and interpretable representation for talking head modeling that we refer to as the Audio-Plane. This factorized design allows for efficient and fine-grained audio-aware spatial encoding, and significantly enhances the model's ability to capture complex lip dynamics driven by speech signals. To further improve region-specific motion modeling, we introduce an audio-guided saliency splatting mechanism based on region-aware modulation, which adaptively emphasizes highly dynamic regions such as the mouth area. This allows the model to focus its learning capacity on where it matters most for accurate speech-driven animation. Extensive experiments on both the self-driven and the cross-driven settings demonstrate that our method achieves state-of-the-art visual quality, precise audio-lip synchronization, and real-time performance, outperforming prior approaches across both 2D- and 3D-based paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21520v1">MADrive: Memory-Augmented Driving Scene Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Recent advances in scene reconstruction have pushed toward highly realistic modeling of autonomous driving (AD) environments using 3D Gaussian splatting. However, the resulting reconstructions remain closely tied to the original observations and struggle to support photorealistic synthesis of significantly altered or novel driving scenarios. This work introduces MADrive, a memory-augmented reconstruction framework designed to extend the capabilities of existing scene reconstruction methods by replacing observed vehicles with visually similar 3D assets retrieved from a large-scale external memory bank. Specifically, we release MAD-Cars, a curated dataset of ${\sim}70$K 360{\deg} car videos captured in the wild and present a retrieval module that finds the most similar car instances in the memory bank, reconstructs the corresponding 3D assets from video, and integrates them into the target scene through orientation alignment and relighting. The resulting replacements provide complete multi-view representations of vehicles in the scene, enabling photorealistic synthesis of substantially altered configurations, as demonstrated in our experiments. Project page: https://yandex-research.github.io/madrive/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21420v1">EndoFlow-SLAM: Real-Time Endoscopic SLAM with Flow-Constrained Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Efficient three-dimensional reconstruction and real-time visualization are critical in surgical scenarios such as endoscopy. In recent years, 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in efficient 3D reconstruction and rendering. Most 3DGS-based Simultaneous Localization and Mapping (SLAM) methods only rely on the appearance constraints for optimizing both 3DGS and camera poses. However, in endoscopic scenarios, the challenges include photometric inconsistencies caused by non-Lambertian surfaces and dynamic motion from breathing affects the performance of SLAM systems. To address these issues, we additionally introduce optical flow loss as a geometric constraint, which effectively constrains both the 3D structure of the scene and the camera motion. Furthermore, we propose a depth regularisation strategy to mitigate the problem of photometric inconsistencies and ensure the validity of 3DGS depth rendering in endoscopic scenes. In addition, to improve scene representation in the SLAM system, we improve the 3DGS refinement strategy by focusing on viewpoints corresponding to Keyframes with suboptimal rendering quality frames, achieving better rendering results. Extensive experiments on the C3VD static dataset and the StereoMIS dynamic dataset demonstrate that our method outperforms existing state-of-the-art methods in novel view synthesis and pose estimation, exhibiting high performance in both static and dynamic surgical scenes. The source code will be publicly available upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21401v1">Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Code: https://github.com/zhirui-gao/Curve-Gaussian Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      This paper presents an end-to-end framework for reconstructing 3D parametric curves directly from multi-view edge maps. Contrasting with existing two-stage methods that follow a sequential ``edge point cloud reconstruction and parametric curve fitting'' pipeline, our one-stage approach optimizes 3D parametric curves directly from 2D edge maps, eliminating error accumulation caused by the inherent optimization gap between disconnected stages. However, parametric curves inherently lack suitability for rendering-based multi-view optimization, necessitating a complementary representation that preserves their geometric properties while enabling differentiable rendering. We propose a novel bi-directional coupling mechanism between parametric curves and edge-oriented Gaussian components. This tight correspondence formulates a curve-aware Gaussian representation, \textbf{CurveGaussian}, that enables differentiable rendering of 3D curves, allowing direct optimization guided by multi-view evidence. Furthermore, we introduce a dynamically adaptive topology optimization framework during training to refine curve structures through linearization, merging, splitting, and pruning operations. Comprehensive evaluations on the ABC dataset and real-world benchmarks demonstrate our one-stage method's superiority over two-stage alternatives, particularly in producing cleaner and more robust reconstructions. Additionally, by directly optimizing parametric curves, our method significantly reduces the parameter count during training, achieving both higher efficiency and superior performance compared to existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21152v1">Geometry and Perception Guided Gaussians for Multiview-consistent 3D Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Generating realistic 3D objects from single-view images requires natural appearance, 3D consistency, and the ability to capture multiple plausible interpretations of unseen regions. Existing approaches often rely on fine-tuning pretrained 2D diffusion models or directly generating 3D information through fast network inference or 3D Gaussian Splatting, but their results generally suffer from poor multiview consistency and lack geometric detail. To takle these issues, we present a novel method that seamlessly integrates geometry and perception priors without requiring additional model training to reconstruct detailed 3D objects from a single image. Specifically, we train three different Gaussian branches initialized from the geometry prior, perception prior and Gaussian noise, respectively. The geometry prior captures the rough 3D shapes, while the perception prior utilizes the 2D pretrained diffusion model to enhance multiview information. Subsequently, we refine 3D Gaussian branches through mutual interaction between geometry and perception priors, further enhanced by a reprojection-based strategy that enforces depth consistency. Experiments demonstrate the higher-fidelity reconstruction results of our method, outperforming existing methods on novel view synthesis and 3D reconstruction, demonstrating robust and consistent 3D object generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21117v1">CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 ICCV 2025, Project Page: https://cl-splats.github.io
    </div>
    <details class="paper-abstract">
      In dynamic 3D environments, accurately updating scene representations over time is crucial for applications in robotics, mixed reality, and embodied AI. As scenes evolve, efficient methods to incorporate changes are needed to maintain up-to-date, high-quality reconstructions without the computational overhead of re-optimizing the entire scene. This paper introduces CL-Splats, which incrementally updates Gaussian splatting-based 3D representations from sparse scene captures. CL-Splats integrates a robust change-detection module that segments updated and static components within the scene, enabling focused, local optimization that avoids unnecessary re-computation. Moreover, CL-Splats supports storing and recovering previous scene states, facilitating temporal segmentation and new scene-analysis applications. Our extensive experiments demonstrate that CL-Splats achieves efficient updates with improved reconstruction quality over the state-of-the-art. This establishes a robust foundation for future real-time adaptation in 3D scene reconstruction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02751v2">RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 ICCV 2025. Project page: https://fcyycf.github.io/RobustSplat/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling. However, existing methods struggle with accurately modeling scenes affected by transient objects, leading to artifacts in the rendered images. We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances. To address this, we propose RobustSplat, a robust solution based on two critical designs. First, we introduce a delayed Gaussian growth strategy that prioritizes optimizing static scene structure before allowing Gaussian splitting/cloning, mitigating overfitting to transient objects in early optimization. Second, we design a scale-cascaded mask bootstrapping approach that first leverages lower-resolution feature similarity supervision for reliable initial transient mask estimation, taking advantage of its stronger semantic consistency and robustness to noise, and then progresses to high-resolution supervision to achieve more precise mask prediction. Extensive experiments on multiple challenging datasets show that our method outperforms existing methods, clearly demonstrating the robustness and effectiveness of our method. Our project page is https://fcyycf.github.io/RobustSplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16606v2">HUG: Hierarchical Urban Gaussian Splatting with Block-Based Reconstruction for Large-Scale Aerial Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 An improved version has recently been accepted to ICCV, manuscript, not camera-ready
    </div>
    <details class="paper-abstract">
      3DGS is an emerging and increasingly popular technology in the field of novel view synthesis. Its highly realistic rendering quality and real-time rendering capabilities make it promising for various applications. However, when applied to large-scale aerial urban scenes, 3DGS methods suffer from issues such as excessive memory consumption, slow training times, prolonged partitioning processes, and significant degradation in rendering quality due to the increased data volume. To tackle these challenges, we introduce \textbf{HUG}, a novel approach that enhances data partitioning and reconstruction quality by leveraging a hierarchical neural Gaussian representation. We first propose a visibility-based data partitioning method that is simple yet highly efficient, significantly outperforming existing methods in speed. Then, we introduce a novel hierarchical weighted training approach, combined with other optimization strategies, to substantially improve reconstruction quality. Our method achieves state-of-the-art results on one synthetic dataset and four real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21009v1">User-in-the-Loop View Sampling with Error Peaking Visualization</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Accepted at IEEE ICIP 2025, Project Page: https://mediated-reality.github.io/projects/yasunaga_icip25/
    </div>
    <details class="paper-abstract">
      Augmented reality (AR) provides ways to visualize missing view samples for novel view synthesis. Existing approaches present 3D annotations for new view samples and task users with taking images by aligning the AR display. This data collection task is known to be mentally demanding and limits capture areas to pre-defined small areas due to the ideal but restrictive underlying sampling theory. To free users from 3D annotations and limited scene exploration, we propose using locally reconstructed light fields and visualizing errors to be removed by inserting new views. Our results show that the error-peaking visualization is less invasive, reduces disappointment in final results, and is satisfactory with fewer view samples in our mobile view synthesis system. We also show that our approach can contribute to recent radiance field reconstruction for larger scenes, such as 3D Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20998v1">DBMovi-GS: Dynamic View Synthesis from Blurry Monocular Video via Sparse-Controlled Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 CVPRW 2025, Neural Fields Beyond Conventional Cameras
    </div>
    <details class="paper-abstract">
      Novel view synthesis is a task of generating scenes from unseen perspectives; however, synthesizing dynamic scenes from blurry monocular videos remains an unresolved challenge that has yet to be effectively addressed. Existing novel view synthesis methods are often constrained by their reliance on high-resolution images or strong assumptions about static geometry and rigid scene priors. Consequently, their approaches lack robustness in real-world environments with dynamic object and camera motion, leading to instability and degraded visual fidelity. To address this, we propose Motion-aware Dynamic View Synthesis from Blurry Monocular Video via Sparse-Controlled Gaussian Splatting (DBMovi-GS), a method designed for dynamic view synthesis from blurry monocular videos. Our model generates dense 3D Gaussians, restoring sharpness from blurry videos and reconstructing detailed 3D geometry of the scene affected by dynamic motion variations. Our model achieves robust performance in novel view synthesis under dynamic blurry scenes and sets a new benchmark in realistic novel view synthesis for blurry monocular video inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11579v2">SweepEvGS: Event-Based 3D Gaussian Splatting for Macro and Micro Radiance Field Rendering from a Single Sweep</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3D-GS) have demonstrated the potential of using 3D Gaussian primitives for high-speed, high-fidelity, and cost-efficient novel view synthesis from continuously calibrated input views. However, conventional methods require high-frame-rate dense and high-quality sharp images, which are time-consuming and inefficient to capture, especially in dynamic environments. Event cameras, with their high temporal resolution and ability to capture asynchronous brightness changes, offer a promising alternative for more reliable scene reconstruction without motion blur. In this paper, we propose SweepEvGS, a novel hardware-integrated method that leverages event cameras for robust and accurate novel view synthesis across various imaging settings from a single sweep. SweepEvGS utilizes the initial static frame with dense event streams captured during a single camera sweep to effectively reconstruct detailed scene views. We also introduce different real-world hardware imaging systems for real-world data collection and evaluation for future research. We validate the robustness and efficiency of SweepEvGS through experiments in three different imaging settings: synthetic objects, real-world macro-level, and real-world micro-level view synthesis. Our results demonstrate that SweepEvGS surpasses existing methods in visual rendering quality, rendering speed, and computational efficiency, highlighting its potential for dynamic practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14384v4">Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 ICCV 2025; A novel one-stage 3DGS-based diffusion for 3D object generation and scene reconstruction from a single view in ~6 seconds
    </div>
    <details class="paper-abstract">
      Existing feedforward image-to-3D methods mainly rely on 2D multi-view diffusion models that cannot guarantee 3D consistency. These methods easily collapse when changing the prompt view direction and mainly handle object-centric cases. In this paper, we propose a novel single-stage 3D diffusion model, DiffusionGS, for object generation and scene reconstruction from a single view. DiffusionGS directly outputs 3D Gaussian point clouds at each timestep to enforce view consistency and allow the model to generate robustly given prompt views of any directions, beyond object-centric inputs. Plus, to improve the capability and generality of DiffusionGS, we scale up 3D training data by developing a scene-object mixed training strategy. Experiments show that DiffusionGS yields improvements of 2.20 dB/23.25 and 1.34 dB/19.16 in PSNR/FID for objects and scenes than the state-of-the-art methods, without depth estimator. Plus, our method enjoys over 5$\times$ faster speed ($\sim$6s on an A100 GPU). Our Project page at https://caiyuanhao1998.github.io/project/DiffusionGS/ shows the video and interactive results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01402v3">ULSR-GS: Ultra Large-scale Surface Reconstruction Gaussian Splatting with Multi-View Geometric Consistency</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Project page: https://ulsrgs.github.io
    </div>
    <details class="paper-abstract">
      While Gaussian Splatting (GS) demonstrates efficient and high-quality scene rendering and small area surface extraction ability, it falls short in handling large-scale aerial image surface extraction tasks. To overcome this, we present ULSR-GS, a framework dedicated to high-fidelity surface extraction in ultra-large-scale scenes, addressing the limitations of existing GS-based mesh extraction methods. Specifically, we propose a point-to-photo partitioning approach combined with a multi-view optimal view matching principle to select the best training images for each sub-region. Additionally, during training, ULSR-GS employs a densification strategy based on multi-view geometric consistency to enhance surface extraction details. Experimental results demonstrate that ULSR-GS outperforms other state-of-the-art GS-based works on large-scale aerial photogrammetry benchmark datasets, significantly improving surface extraction accuracy in complex urban environments. Project page: https://ulsrgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01109v2">FGS-SLAM: Fourier-based Gaussian Splatting for Real-time SLAM with Sparse and Dense Map Fusion</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      3D gaussian splatting has advanced simultaneous localization and mapping (SLAM) technology by enabling real-time positioning and the construction of high-fidelity maps. However, the uncertainty in gaussian position and initialization parameters introduces challenges, often requiring extensive iterative convergence and resulting in redundant or insufficient gaussian representations. To address this, we introduce a novel adaptive densification method based on Fourier frequency domain analysis to establish gaussian priors for rapid convergence. Additionally, we propose constructing independent and unified sparse and dense maps, where a sparse map supports efficient tracking via Generalized Iterative Closest Point (GICP) and a dense map creates high-fidelity visual representations. This is the first SLAM system leveraging frequency domain analysis to achieve high-quality gaussian mapping in real-time. Experimental results demonstrate an average frame rate of 36 FPS on Replica and TUM RGB-D datasets, achieving competitive accuracy in both localization and mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20202v1">RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      With the advancement of Gaussian Splatting techniques, a growing number of datasets based on this representation have been developed. However, performing accurate and efficient clipping for Gaussian Splatting remains a challenging and unresolved problem, primarily due to the volumetric nature of Gaussian primitives, which makes hard clipping incapable of precisely localizing their pixel-level contributions. In this paper, we propose a hybrid rendering framework that combines rasterization and ray tracing to achieve efficient and high-fidelity clipping of Gaussian Splatting data. At the core of our method is the RaRa strategy, which first leverages rasterization to quickly identify Gaussians intersected by the clipping plane, followed by ray tracing to compute attenuation weights based on their partial occlusion. These weights are then used to accurately estimate each Gaussian's contribution to the final image, enabling smooth and continuous clipping effects. We validate our approach on diverse datasets, including general Gaussians, hair strand Gaussians, and multi-layer Gaussians, and conduct user studies to evaluate both perceptual quality and quantitative performance. Experimental results demonstrate that our method delivers visually superior results while maintaining real-time rendering performance and preserving high fidelity in the unclipped regions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16767v4">ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Project page: https://liuff19.github.io/ReconX
    </div>
    <details class="paper-abstract">
      Advancements in 3D scene reconstruction have transformed 2D images from the real world into 3D models, producing realistic 3D results from hundreds of input photos. Despite great success in dense-view reconstruction scenarios, rendering a detailed scene from insufficient captured views is still an ill-posed optimization problem, often resulting in artifacts and distortions in unseen areas. In this paper, we propose ReconX, a novel 3D scene reconstruction paradigm that reframes the ambiguous reconstruction challenge as a temporal generation task. The key insight is to unleash the strong generative prior of large pre-trained video diffusion models for sparse-view reconstruction. However, 3D view consistency struggles to be accurately preserved in directly generated video frames from pre-trained models. To address this, given limited input views, the proposed ReconX first constructs a global point cloud and encodes it into a contextual space as the 3D structure condition. Guided by the condition, the video diffusion model then synthesizes video frames that are both detail-preserved and exhibit a high degree of 3D consistency, ensuring the coherence of the scene from various perspectives. Finally, we recover the 3D scene from the generated video through a confidence-aware 3D Gaussian Splatting optimization scheme. Extensive experiments on various real-world datasets show the superiority of our ReconX over state-of-the-art methods in terms of quality and generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10504v2">USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Spike cameras, as an innovative neuromorphic camera that captures scenes with the 0-1 bit stream at 40 kHz, are increasingly employed for the 3D reconstruction task via Neural Radiance Fields (NeRF) or 3D Gaussian Splatting (3DGS). Previous spike-based 3D reconstruction approaches often employ a casecased pipeline: starting with high-quality image reconstruction from spike streams based on established spike-to-image reconstruction algorithms, then progressing to camera pose estimation and 3D reconstruction. However, this cascaded approach suffers from substantial cumulative errors, where quality limitations of initial image reconstructions negatively impact pose estimation, ultimately degrading the fidelity of the 3D reconstruction. To address these issues, we propose a synergistic optimization framework, \textbf{USP-Gaussian}, that unifies spike-based image reconstruction, pose correction, and Gaussian splatting into an end-to-end framework. Leveraging the multi-view consistency afforded by 3DGS and the motion capture capability of the spike camera, our framework enables a joint iterative optimization that seamlessly integrates information between the spike-to-image network and 3DGS. Experiments on synthetic datasets with accurate poses demonstrate that our method surpasses previous approaches by effectively eliminating cascading errors. Moreover, we integrate pose optimization to achieve robust 3D reconstruction in real-world scenarios with inaccurate initial poses, outperforming alternative methods by effectively reducing noise and preserving fine texture details. Our code, data and trained models will be available at https://github.com/chenkang455/USP-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20875v1">3DGH: 3D Head Generation with Composable Hair and Face</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted to SIGGRAPH 2025. Project page: https://c-he.github.io/projects/3dgh/
    </div>
    <details class="paper-abstract">
      We present 3DGH, an unconditional generative model for 3D human heads with composable hair and face components. Unlike previous work that entangles the modeling of hair and face, we propose to separate them using a novel data representation with template-based 3D Gaussian Splatting, in which deformable hair geometry is introduced to capture the geometric variations across different hairstyles. Based on this data representation, we design a 3D GAN-based architecture with dual generators and employ a cross-attention mechanism to model the inherent correlation between hair and face. The model is trained on synthetic renderings using carefully designed objectives to stabilize training and facilitate hair-face separation. We conduct extensive experiments to validate the design choice of 3DGH, and evaluate it both qualitatively and quantitatively by comparing with several state-of-the-art 3D GAN methods, demonstrating its effectiveness in unconditional full-head image synthesis and composable 3D hairstyle editing. More details will be available on our project page: https://c-he.github.io/projects/3dgh/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21633v1">SAR-GS: 3D Gaussian Splatting for Synthetic Aperture Radar Target Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Three-dimensional target reconstruction from synthetic aperture radar (SAR) imagery is crucial for interpreting complex scattering information in SAR data. However, the intricate electromagnetic scattering mechanisms inherent to SAR imaging pose significant reconstruction challenges. Inspired by the remarkable success of 3D Gaussian Splatting (3D-GS) in optical domain reconstruction, this paper presents a novel SAR Differentiable Gaussian Splatting Rasterizer (SDGR) specifically designed for SAR target reconstruction. Our approach combines Gaussian splatting with the Mapping and Projection Algorithm to compute scattering intensities of Gaussian primitives and generate simulated SAR images through SDGR. Subsequently, the loss function between the rendered image and the ground truth image is computed to optimize the Gaussian primitive parameters representing the scene, while a custom CUDA gradient flow is employed to replace automatic differentiation for accelerated gradient computation. Through experiments involving the rendering of simplified architectural targets and SAR images of multiple vehicle targets, we validate the imaging rationality of SDGR on simulated SAR imagery. Furthermore, the effectiveness of our method for target reconstruction is demonstrated on both simulated and real-world datasets containing multiple vehicle targets, with quantitative evaluations conducted to assess its reconstruction performance. Experimental results indicate that our approach can effectively reconstruct the geometric structures and scattering properties of targets, thereby providing a novel solution for 3D reconstruction in the field of SAR imaging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21632v1">SkinningGS: Editable Dynamic Human Scene Reconstruction Using Gaussian Splatting Based on a Skinning Model</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Reconstructing an interactive human avatar and the background from a monocular video of a dynamic human scene is highly challenging. In this work we adopt a strategy of point cloud decoupling and joint optimization to achieve the decoupled reconstruction of backgrounds and human bodies while preserving the interactivity of human motion. We introduce a position texture to subdivide the Skinned Multi-Person Linear (SMPL) body model's surface and grow the human point cloud. To capture fine details of human dynamics and deformations, we incorporate a convolutional neural network structure to predict human body point cloud features based on texture. This strategy makes our approach free of hyperparameter tuning for densification and efficiently represents human points with half the point cloud of HUGS. This approach ensures high-quality human reconstruction and reduces GPU resource consumption during training. As a result, our method surpasses the previous state-of-the-art HUGS in reconstruction metrics while maintaining the ability to generalize to novel poses and views. Furthermore, our technique achieves real-time rendering at over 100 FPS, $\sim$6$\times$ the HUGS speed using only Linear Blend Skinning (LBS) weights for human transformation. Additionally, this work demonstrates that this framework can be extended to animal scene reconstruction when an accurately-posed model of an animal is available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19842v1">ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Model</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Multi-task robotic bimanual manipulation is becoming increasingly popular as it enables sophisticated tasks that require diverse dual-arm collaboration patterns. Compared to unimanual manipulation, bimanual tasks pose challenges to understanding the multi-body spatiotemporal dynamics. An existing method ManiGaussian pioneers encoding the spatiotemporal dynamics into the visual representation via Gaussian world model for single-arm settings, which ignores the interaction of multiple embodiments for dual-arm systems with significant performance drop. In this paper, we propose ManiGaussian++, an extension of ManiGaussian framework that improves multi-task bimanual manipulation by digesting multi-body scene dynamics through a hierarchical Gaussian world model. To be specific, we first generate task-oriented Gaussian Splatting from intermediate visual features, which aims to differentiate acting and stabilizing arms for multi-body spatiotemporal dynamics modeling. We then build a hierarchical Gaussian world model with the leader-follower architecture, where the multi-body spatiotemporal dynamics is mined for intermediate visual representation via future scene prediction. The leader predicts Gaussian Splatting deformation caused by motions of the stabilizing arm, through which the follower generates the physical consequences resulted from the movement of the acting arm. As a result, our method significantly outperforms the current state-of-the-art bimanual manipulation techniques by an improvement of 20.2% in 10 simulated tasks, and achieves 60% success rate on average in 9 challenging real-world tasks. Our code is available at https://github.com/April-Yz/ManiGaussian_Bimanual.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15242v2">RA-NeRF: Robust Neural Radiance Field Reconstruction with Accurate Camera Pose Estimation under Complex Trajectories</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 IROS 2025
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have emerged as powerful tools for 3D reconstruction and SLAM tasks. However, their performance depends heavily on accurate camera pose priors. Existing approaches attempt to address this issue by introducing external constraints but fall short of achieving satisfactory accuracy, particularly when camera trajectories are complex. In this paper, we propose a novel method, RA-NeRF, capable of predicting highly accurate camera poses even with complex camera trajectories. Following the incremental pipeline, RA-NeRF reconstructs the scene using NeRF with photometric consistency and incorporates flow-driven pose regulation to enhance robustness during initialization and localization. Additionally, RA-NeRF employs an implicit pose filter to capture the camera movement pattern and eliminate the noise for pose estimation. To validate our method, we conduct extensive experiments on the Tanks\&Temple dataset for standard evaluation, as well as the NeRFBuster dataset, which presents challenging camera pose trajectories. On both datasets, RA-NeRF achieves state-of-the-art results in both camera pose estimation and visual quality, demonstrating its effectiveness and robustness in scene reconstruction under complex pose trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00814v5">VR-Doh: Hands-on 3D Modeling in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      We introduce VR-Doh, an open-source, hands-on 3D modeling system that enables intuitive creation and manipulation of elastoplastic objects in Virtual Reality (VR). By customizing the Material Point Method (MPM) for real-time simulation of hand-induced large deformations and enhancing 3D Gaussian Splatting for seamless rendering, VR-Doh provides an interactive and immersive 3D modeling experience. Users can naturally sculpt, deform, and edit objects through both contact- and gesture-based hand-object interactions. To achieve real-time performance, our system incorporates localized simulation techniques, particle-level collision handling, and the decoupling of physical and appearance representations, ensuring smooth and responsive interactions. VR-Doh supports both object creation and editing, enabling diverse modeling tasks such as designing food items, characters, and interlocking structures, all resulting in simulation-ready assets. User studies with both novice and experienced participants highlight the system's intuitive design, immersive feedback, and creative potential. Compared to existing geometric modeling tools, VR-Doh offers enhanced accessibility and natural interaction, making it a powerful tool for creative exploration in VR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16767v3">ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Project page: https://liuff19.github.io/ReconX
    </div>
    <details class="paper-abstract">
      Advancements in 3D scene reconstruction have transformed 2D images from the real world into 3D models, producing realistic 3D results from hundreds of input photos. Despite great success in dense-view reconstruction scenarios, rendering a detailed scene from insufficient captured views is still an ill-posed optimization problem, often resulting in artifacts and distortions in unseen areas. In this paper, we propose ReconX, a novel 3D scene reconstruction paradigm that reframes the ambiguous reconstruction challenge as a temporal generation task. The key insight is to unleash the strong generative prior of large pre-trained video diffusion models for sparse-view reconstruction. However, 3D view consistency struggles to be accurately preserved in directly generated video frames from pre-trained models. To address this, given limited input views, the proposed ReconX first constructs a global point cloud and encodes it into a contextual space as the 3D structure condition. Guided by the condition, the video diffusion model then synthesizes video frames that are both detail-preserved and exhibit a high degree of 3D consistency, ensuring the coherence of the scene from various perspectives. Finally, we recover the 3D scene from the generated video through a confidence-aware 3D Gaussian Splatting optimization scheme. Extensive experiments on various real-world datasets show the superiority of our ReconX over state-of-the-art methods in terms of quality and generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07494v4">SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 IROS 2025
    </div>
    <details class="paper-abstract">
      We propose SemGauss-SLAM, a dense semantic SLAM system utilizing 3D Gaussian representation, that enables accurate 3D semantic mapping, robust camera tracking, and high-quality rendering simultaneously. In this system, we incorporate semantic feature embedding into 3D Gaussian representation, which effectively encodes semantic information within the spatial layout of the environment for precise semantic scene representation. Furthermore, we propose feature-level loss for updating 3D Gaussian representation, enabling higher-level guidance for 3D Gaussian optimization. In addition, to reduce cumulative drift in tracking and improve semantic reconstruction accuracy, we introduce semantic-informed bundle adjustment. By leveraging multi-frame semantic associations, this strategy enables joint optimization of 3D Gaussian representation and camera poses, resulting in low-drift tracking and accurate semantic mapping. Our SemGauss-SLAM demonstrates superior performance over existing radiance field-based SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in high-precision semantic segmentation and dense semantic mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19415v1">Virtual Memory for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Based on the Master Thesis from Jonathan Haberl from 2024, Submitted to TVCG in Feb. 2025;
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting represents a breakthrough in the field of novel view synthesis. It establishes Gaussians as core rendering primitives for highly accurate real-world environment reconstruction. Recent advances have drastically increased the size of scenes that can be created. In this work, we present a method for rendering large and complex 3D Gaussian Splatting scenes using virtual memory. By leveraging well-established virtual memory and virtual texturing techniques, our approach efficiently identifies visible Gaussians and dynamically streams them to the GPU just in time for real-time rendering. Selecting only the necessary Gaussians for both storage and rendering results in reduced memory usage and effectively accelerates rendering, especially for highly complex scenes. Furthermore, we demonstrate how level of detail can be integrated into our proposed method to further enhance rendering speed for large-scale scenes. With an optimized implementation, we highlight key practical considerations and thoroughly evaluate the proposed technique and its impact on desktop and mobile devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19291v1">HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      We propose HoliGS, a novel deformable Gaussian splatting framework that addresses embodied view synthesis from long monocular RGB videos. Unlike prior 4D Gaussian splatting and dynamic NeRF pipelines, which struggle with training overhead in minute-long captures, our method leverages invertible Gaussian Splatting deformation networks to reconstruct large-scale, dynamic environments accurately. Specifically, we decompose each scene into a static background plus time-varying objects, each represented by learned Gaussian primitives undergoing global rigid transformations, skeleton-driven articulation, and subtle non-rigid deformations via an invertible neural flow. This hierarchical warping strategy enables robust free-viewpoint novel-view rendering from various embodied camera trajectories by attaching Gaussians to a complete canonical foreground shape (\eg, egocentric or third-person follow), which may involve substantial viewpoint changes and interactions between multiple actors. Our experiments demonstrate that \ourmethod~ achieves superior reconstruction quality on challenging datasets while significantly reducing both training and rendering time compared to state-of-the-art monocular deformable NeRFs. These results highlight a practical and scalable solution for EVS in real-world scenarios. The source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21629v1">ICP-3DGS: SfM-free 3D Gaussian Splatting for Large-scale Unbounded Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 6 pages, Source code is available at https://github.com/Chenhao-Z/ICP-3DGS. To appear at ICIP 2025
    </div>
    <details class="paper-abstract">
      In recent years, neural rendering methods such as NeRFs and 3D Gaussian Splatting (3DGS) have made significant progress in scene reconstruction and novel view synthesis. However, they heavily rely on preprocessed camera poses and 3D structural priors from structure-from-motion (SfM), which are challenging to obtain in outdoor scenarios. To address this challenge, we propose to incorporate Iterative Closest Point (ICP) with optimization-based refinement to achieve accurate camera pose estimation under large camera movements. Additionally, we introduce a voxel-based scene densification approach to guide the reconstruction in large-scale scenes. Experiments demonstrate that our approach ICP-3DGS outperforms existing methods in both camera pose estimation and novel view synthesis across indoor and outdoor scenes of various scales. Source code is available at https://github.com/Chenhao-Z/ICP-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18885v1">GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has emerged as an expressive scene representation for RGB-D visual SLAM, but its application to large-scale, multi-agent outdoor environments remains unexplored. Multi-agent Gaussian SLAM is a promising approach to rapid exploration and reconstruction of environments, offering scalable environment representations, but existing approaches are limited to small-scale, indoor environments. To that end, we propose Gaussian Reconstruction via Multi-Agent Dense SLAM, or GRAND-SLAM, a collaborative Gaussian splatting SLAM method that integrates i) an implicit tracking module based on local optimization over submaps and ii) an approach to inter- and intra-robot loop closure integrated into a pose-graph optimization framework. Experiments show that GRAND-SLAM provides state-of-the-art tracking performance and 28% higher PSNR than existing methods on the Replica indoor dataset, as well as 91% lower multi-agent tracking error and improved rendering over existing multi-agent methods on the large-scale, outdoor Kimera-Multi dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17590v2">CGS-GAN: 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Main paper 12 pages, supplementary materials 8 pages
    </div>
    <details class="paper-abstract">
      Recently, 3D GANs based on 3D Gaussian splatting have been proposed for high quality synthesis of human heads. However, existing methods stabilize training and enhance rendering quality from steep viewpoints by conditioning the random latent vector on the current camera position. This compromises 3D consistency, as we observe significant identity changes when re-synthesizing the 3D head with each camera shift. Conversely, fixing the camera to a single viewpoint yields high-quality renderings for that perspective but results in poor performance for novel views. Removing view-conditioning typically destabilizes GAN training, often causing the training to collapse. In response to these challenges, we introduce CGS-GAN, a novel 3D Gaussian Splatting GAN framework that enables stable training and high-quality 3D-consistent synthesis of human heads without relying on view-conditioning. To ensure training stability, we introduce a multi-view regularization technique that enhances generator convergence with minimal computational overhead. Additionally, we adapt the conditional loss used in existing 3D Gaussian splatting GANs and propose a generator architecture designed to not only stabilize training but also facilitate efficient rendering and straightforward scaling, enabling output resolutions up to $2048^2$. To evaluate the capabilities of CGS-GAN, we curate a new dataset derived from FFHQ. This dataset enables very high resolutions, focuses on larger portions of the human head, reduces view-dependent artifacts for improved 3D consistency, and excludes images where subjects are obscured by hands or other objects. As a result, our approach achieves very high rendering quality, supported by competitive FID scores, while ensuring consistent 3D scene generation. Check our our project page here: https://fraunhoferhhi.github.io/cgs-gan/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18792v1">ViDAR: Video Diffusion-Aware 4D Reconstruction From Monocular Inputs</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Dynamic Novel View Synthesis aims to generate photorealistic views of moving subjects from arbitrary viewpoints. This task is particularly challenging when relying on monocular video, where disentangling structure from motion is ill-posed and supervision is scarce. We introduce Video Diffusion-Aware Reconstruction (ViDAR), a novel 4D reconstruction framework that leverages personalised diffusion models to synthesise a pseudo multi-view supervision signal for training a Gaussian splatting representation. By conditioning on scene-specific features, ViDAR recovers fine-grained appearance details while mitigating artefacts introduced by monocular ambiguity. To address the spatio-temporal inconsistency of diffusion-based supervision, we propose a diffusion-aware loss function and a camera pose optimisation strategy that aligns synthetic views with the underlying scene geometry. Experiments on DyCheck, a challenging benchmark with extreme viewpoint variation, show that ViDAR outperforms all state-of-the-art baselines in visual quality and geometric consistency. We further highlight ViDAR's strong improvement over baselines on dynamic regions and provide a new benchmark to compare performance in reconstructing motion-rich parts of the scene. Project page: https://vidar-4d.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18787v1">3D Arena: An Open Platform for Generative 3D Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Evaluating Generative 3D models remains challenging due to misalignment between automated metrics and human perception of quality. Current benchmarks rely on image-based metrics that ignore 3D structure or geometric measures that fail to capture perceptual appeal and real-world utility. To address this gap, we present 3D Arena, an open platform for evaluating image-to-3D generation models through large-scale human preference collection using pairwise comparisons. Since launching in June 2024, the platform has collected 123,243 votes from 8,096 users across 19 state-of-the-art models, establishing the largest human preference evaluation for Generative 3D. We contribute the iso3d dataset of 100 evaluation prompts and demonstrate quality control achieving 99.75% user authenticity through statistical fraud detection. Our ELO-based ranking system provides reliable model assessment, with the platform becoming an established evaluation resource. Through analysis of this preference data, we present insights into human preference patterns. Our findings reveal preferences for visual presentation features, with Gaussian splat outputs achieving a 16.6 ELO advantage over meshes and textured models receiving a 144.1 ELO advantage over untextured models. We provide recommendations for improving evaluation methods, including multi-criteria assessment, task-oriented evaluation, and format-aware comparison. The platform's community engagement establishes 3D Arena as a benchmark for the field while advancing understanding of human-centered evaluation in Generative 3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18677v1">Reconstructing Tornadoes in 3D with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-23
    </div>
    <details class="paper-abstract">
      Accurately reconstructing the 3D structure of tornadoes is critically important for understanding and preparing for this highly destructive weather phenomenon. While modern 3D scene reconstruction techniques, such as 3D Gaussian splatting (3DGS), could provide a valuable tool for reconstructing the 3D structure of tornados, at present we are critically lacking a controlled tornado dataset with which to develop and validate these tools. In this work we capture and release a novel multiview dataset of a small lab-based tornado. We demonstrate one can effectively reconstruct and visualize the 3D structure of this tornado using 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16262v2">R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 Please visit our project page at https://github.com/CMLab-Korea/Awesome-3D-Low-Level-Vision
    </div>
    <details class="paper-abstract">
      Neural rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved significant progress in photorealistic 3D scene reconstruction and novel view synthesis. However, most existing models assume clean and high-resolution (HR) multi-view inputs, which limits their robustness under real-world degradations such as noise, blur, low-resolution (LR), and weather-induced artifacts. To address these limitations, the emerging field of 3D Low-Level Vision (3D LLV) extends classical 2D Low-Level Vision tasks including super-resolution (SR), deblurring, weather degradation removal, restoration, and enhancement into the 3D spatial domain. This survey, referred to as R\textsuperscript{3}eVision, provides a comprehensive overview of robust rendering, restoration, and enhancement for 3D LLV by formalizing the degradation-aware rendering problem and identifying key challenges related to spatio-temporal consistency and ill-posed optimization. Recent methods that integrate LLV into neural rendering frameworks are categorized to illustrate how they enable high-fidelity 3D reconstruction under adverse conditions. Application domains such as autonomous driving, AR/VR, and robotics are also discussed, where reliable 3D perception from degraded inputs is critical. By reviewing representative methods, datasets, and evaluation protocols, this work positions 3D LLV as a fundamental direction for robust 3D content generation and scene-level reconstruction in real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14249v2">CLIP-GS: CLIP-Informed Gaussian Splatting for View-Consistent 3D Indoor Semantic Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 ACM TOMM 2025
    </div>
    <details class="paper-abstract">
      Exploiting 3D Gaussian Splatting (3DGS) with Contrastive Language-Image Pre-Training (CLIP) models for open-vocabulary 3D semantic understanding of indoor scenes has emerged as an attractive research focus. Existing methods typically attach high-dimensional CLIP semantic embeddings to 3D Gaussians and leverage view-inconsistent 2D CLIP semantics as Gaussian supervision, resulting in efficiency bottlenecks and deficient 3D semantic consistency. To address these challenges, we present CLIP-GS, efficiently achieving a coherent semantic understanding of 3D indoor scenes via the proposed Semantic Attribute Compactness (SAC) and 3D Coherent Regularization (3DCR). SAC approach exploits the naturally unified semantics within objects to learn compact, yet effective, semantic Gaussian representations, enabling highly efficient rendering (>100 FPS). 3DCR enforces semantic consistency in 2D and 3D domains: In 2D, 3DCR utilizes refined view-consistent semantic outcomes derived from 3DGS to establish cross-view coherence constraints; in 3D, 3DCR encourages features similar among 3D Gaussian primitives associated with the same object, leading to more precise and coherent segmentation results. Extensive experimental results demonstrate that our method remarkably suppresses existing state-of-the-art approaches, achieving mIoU improvements of 21.20% and 13.05% on ScanNet and Replica datasets, respectively, while maintaining real-time rendering speed. Furthermore, our approach exhibits superior performance even with sparse input data, substantiating its robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14135v2">GAF: Gaussian Action Field as a Dynamic World Model for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-06-23
      | 💬 http://chaiying1.github.io/GAF.github.io/project_page/
    </div>
    <details class="paper-abstract">
      Accurate action inference is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we propose a Vision-to-4D-to-Action (V-4D-A) framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing simultaneous modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF supports three key query types: reconstruction of the current scene, prediction of future frames, and estimation of initial action via robot motion. Furthermore, the high-quality current and future frames generated by GAF facilitate manipulation action refinement through a GAF-guided diffusion model. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average success rate in robotic manipulation tasks by 10.33% over state-of-the-art methods. Project page: http://chaiying1.github.io/GAF.github.io/project_page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17636v1">3D Gaussian Splatting for Fine-Detailed Surface Reconstruction in Large-Scale Scene</a></div>
    <div class="paper-meta">
      📅 2025-06-21
      | 💬 IROS 2025
    </div>
    <details class="paper-abstract">
      Recent developments in 3D Gaussian Splatting have made significant advances in surface reconstruction. However, scaling these methods to large-scale scenes remains challenging due to high computational demands and the complex dynamic appearances typical of outdoor environments. These challenges hinder the application in aerial surveying and autonomous driving. This paper proposes a novel solution to reconstruct large-scale surfaces with fine details, supervised by full-sized images. Firstly, we introduce a coarse-to-fine strategy to reconstruct a coarse model efficiently, followed by adaptive scene partitioning and sub-scene refining from image segments. Additionally, we integrate a decoupling appearance model to capture global appearance variations and a transient mask model to mitigate interference from moving objects. Finally, we expand the multi-view constraint and introduce a single-view regularization for texture-less areas. Our experiments were conducted on the publicly available dataset GauU-Scene V2, which was captured using unmanned aerial vehicles. To the best of our knowledge, our method outperforms existing NeRF-based and Gaussian-based methods, achieving high-fidelity visual results and accurate surface from full-size image optimization. Open-source code will be available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17212v1">Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-20
    </div>
    <details class="paper-abstract">
      Articulated objects are common in the real world, yet modeling their structure and motion remains a challenging task for 3D reconstruction methods. In this work, we introduce Part$^{2}$GS, a novel framework for modeling articulated digital twins of multi-part objects with high-fidelity geometry and physically consistent articulation. Part$^{2}$GS leverages a part-aware 3D Gaussian representation that encodes articulated components with learnable attributes, enabling structured, disentangled transformations that preserve high-fidelity geometry. To ensure physically consistent motion, we propose a motion-aware canonical representation guided by physics-based constraints, including contact enforcement, velocity consistency, and vector-field alignment. Furthermore, we introduce a field of repel points to prevent part collisions and maintain stable articulation paths, significantly improving motion coherence over baselines. Extensive evaluations on both synthetic and real-world datasets show that Part$^{2}$GS consistently outperforms state-of-the-art methods by up to 10$\times$ in Chamfer Distance for movable parts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12400v2">Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Accepted to International Conference on Machine Learning (ICML) 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis. However, existing methods struggle to adaptively optimize the distribution of Gaussian primitives based on scene characteristics, making it challenging to balance reconstruction quality and efficiency. Inspired by human perception, we propose scene-adaptive perceptual densification for Gaussian Splatting (Perceptual-GS), a novel framework that integrates perceptual sensitivity into the 3DGS training process to address this challenge. We first introduce a perception-aware representation that models human visual sensitivity while constraining the number of Gaussian primitives. Building on this foundation, we develop a perceptual sensitivity-adaptive distribution to allocate finer Gaussian granularity to visually critical regions, enhancing reconstruction quality and robustness. Extensive evaluations on multiple datasets, including BungeeNeRF for large-scale scenes, demonstrate that Perceptual-GS achieves state-of-the-art performance in reconstruction quality, efficiency, and robustness. The code is publicly available at: https://github.com/eezkni/Perceptual-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13183v3">Real-time Free-view Human Rendering from Sparse-view RGB Videos using Double Unprojected Textures</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Accepted at CVPR 2025, Project page: https://vcai.mpi-inf.mpg.de/projects/DUT/
    </div>
    <details class="paper-abstract">
      Real-time free-view human rendering from sparse-view RGB inputs is a challenging task due to the sensor scarcity and the tight time budget. To ensure efficiency, recent methods leverage 2D CNNs operating in texture space to learn rendering primitives. However, they either jointly learn geometry and appearance, or completely ignore sparse image information for geometry estimation, significantly harming visual quality and robustness to unseen body poses. To address these issues, we present Double Unprojected Textures, which at the core disentangles coarse geometric deformation estimation from appearance synthesis, enabling robust and photorealistic 4K rendering in real-time. Specifically, we first introduce a novel image-conditioned template deformation network, which estimates the coarse deformation of the human template from a first unprojected texture. This updated geometry is then used to apply a second and more accurate texture unprojection. The resulting texture map has fewer artifacts and better alignment with input views, which benefits our learning of finer-level geometry and appearance represented by Gaussian splats. We validate the effectiveness and efficiency of the proposed method in quantitative and qualitative experiments, which significantly surpasses other state-of-the-art methods. Project page: https://vcai.mpi-inf.mpg.de/projects/DUT/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2210.00379v6">NeRF: Neural Radiance Field in 3D Vision: A Comprehensive Review (Updated Post-Gaussian Splatting)</a></div>
    <div class="paper-meta">
      📅 2025-06-20
      | 💬 Updated Post-Gaussian Splatting
    </div>
    <details class="paper-abstract">
      In March 2020, Neural Radiance Field (NeRF) revolutionized Computer Vision, allowing for implicit, neural network-based scene representation and novel view synthesis. NeRF models have found diverse applications in robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, and more. In August 2023, Gaussian Splatting, a direct competitor to the NeRF-based framework, was proposed, gaining tremendous momentum and overtaking NeRF-based research in terms of interest as the dominant framework for novel view synthesis. We present a comprehensive survey of NeRF papers from the past five years (2020-2025). These include papers from the pre-Gaussian Splatting era, where NeRF dominated the field for novel view synthesis and 3D implicit and hybrid representation neural field learning. We also include works from the post-Gaussian Splatting era where NeRF and implicit/hybrid neural fields found more niche applications. Our survey is organized into architecture and application-based taxonomies in the pre-Gaussian Splatting era, as well as a categorization of active research areas for NeRF, neural field, and implicit/hybrid neural representation methods. We provide an introduction to the theory of NeRF and its training via differentiable volume rendering. We also present a benchmark comparison of the performance and speed of classical NeRF, implicit and hybrid neural representation, and neural field models, and an overview of key datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16262v1">R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 Please visit our project page at https://github.com/CMLab-Korea/Awesome-3D-Low-Level-Vision
    </div>
    <details class="paper-abstract">
      Neural rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved significant progress in photorealistic 3D scene reconstruction and novel view synthesis. However, most existing models assume clean and high-resolution (HR) multi-view inputs, which limits their robustness under real-world degradations such as noise, blur, low-resolution (LR), and weather-induced artifacts. To address these limitations, the emerging field of 3D Low-Level Vision (3D LLV) extends classical 2D Low-Level Vision tasks including super-resolution (SR), deblurring, weather degradation removal, restoration, and enhancement into the 3D spatial domain. This survey, referred to as R\textsuperscript{3}eVision, provides a comprehensive overview of robust rendering, restoration, and enhancement for 3D LLV by formalizing the degradation-aware rendering problem and identifying key challenges related to spatio-temporal consistency and ill-posed optimization. Recent methods that integrate LLV into neural rendering frameworks are categorized to illustrate how they enable high-fidelity 3D reconstruction under adverse conditions. Application domains such as autonomous driving, AR/VR, and robotics are also discussed, where reliable 3D perception from degraded inputs is critical. By reviewing representative methods, datasets, and evaluation protocols, this work positions 3D LLV as a fundamental direction for robust 3D content generation and scene-level reconstruction in real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15948v1">Information-computation trade-offs in non-linear transforms</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 Authors listed in alphabetical order of last name
    </div>
    <details class="paper-abstract">
      In this work, we explore the interplay between information and computation in non-linear transform-based compression for broad classes of modern information-processing tasks. We first investigate two emerging nonlinear data transformation frameworks for image compression: Implicit Neural Representations (INRs) and 2D Gaussian Splatting (GS). We analyze their representational properties, behavior under lossy compression, and convergence dynamics. Our results highlight key trade-offs between INR's compact, resolution-flexible neural field representations and GS's highly parallelizable, spatially interpretable fitting, providing insights for future hybrid and compression-aware frameworks. Next, we introduce the textual transform that enables efficient compression at ultra-low bitrate regimes and simultaneously enhances human perceptual satisfaction. When combined with the concept of denoising via lossy compression, the textual transform becomes a powerful tool for denoising tasks. Finally, we present a Lempel-Ziv (LZ78) "transform", a universal method that, when applied to any member of a broad compressor family, produces new compressors that retain the asymptotic universality guarantees of the LZ78 algorithm. Collectively, these three transforms illuminate the fundamental trade-offs between coding efficiency and computational cost. We discuss how these insights extend beyond compression to tasks such as classification, denoising, and generative AI, suggesting new pathways for using non-linear transformations to balance resource constraints and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15680v1">Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Project page: https://kywind.github.io/pgnd
    </div>
    <details class="paper-abstract">
      Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12787v2">Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Modeling the wireless radiance field (WRF) is fundamental to modern communication systems, enabling key tasks such as localization, sensing, and channel estimation. Traditional approaches, which rely on empirical formulas or physical simulations, often suffer from limited accuracy or require strong scene priors. Recent neural radiance field (NeRF-based) methods improve reconstruction fidelity through differentiable volumetric rendering, but their reliance on computationally expensive multilayer perceptron (MLP) queries hinders real-time deployment. To overcome these challenges, we introduce Gaussian splatting (GS) to the wireless domain, leveraging its efficiency in modeling optical radiance fields to enable compact and accurate WRF reconstruction. Specifically, we propose SwiftWRF, a deformable 2D Gaussian splatting framework that synthesizes WRF spectra at arbitrary positions under single-sided transceiver mobility. SwiftWRF employs CUDA-accelerated rasterization to render spectra at over 100000 fps and uses a lightweight MLP to model the deformation of 2D Gaussians, effectively capturing mobility-induced WRF variations. In addition to novel spectrum synthesis, the efficacy of SwiftWRF is further underscored in its applications in angle-of-arrival (AoA) and received signal strength indicator (RSSI) prediction. Experiments conducted on both real-world and synthetic indoor scenes demonstrate that SwiftWRF can reconstruct WRF spectra up to 500x faster than existing state-of-the-art methods, while significantly enhancing its signal quality. The project page is https://evan-sudo.github.io/swiftwrf/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15242v1">RA-NeRF: Robust Neural Radiance Field Reconstruction with Accurate Camera Pose Estimation under Complex Trajectories</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 IROS 2025
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have emerged as powerful tools for 3D reconstruction and SLAM tasks. However, their performance depends heavily on accurate camera pose priors. Existing approaches attempt to address this issue by introducing external constraints but fall short of achieving satisfactory accuracy, particularly when camera trajectories are complex. In this paper, we propose a novel method, RA-NeRF, capable of predicting highly accurate camera poses even with complex camera trajectories. Following the incremental pipeline, RA-NeRF reconstructs the scene using NeRF with photometric consistency and incorporates flow-driven pose regulation to enhance robustness during initialization and localization. Additionally, RA-NeRF employs an implicit pose filter to capture the camera movement pattern and eliminate the noise for pose estimation. To validate our method, we conduct extensive experiments on the Tanks\&Temple dataset for standard evaluation, as well as the NeRFBuster dataset, which presents challenging camera pose trajectories. On both datasets, RA-NeRF achieves state-of-the-art results in both camera pose estimation and visual quality, demonstrating its effectiveness and robustness in scene reconstruction under complex pose trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14742v1">SyncTalk++: High-Fidelity and Efficient Synchronized Talking Heads Synthesis Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Achieving high synchronization in the synthesis of realistic, speech-driven talking head videos presents a significant challenge. A lifelike talking head requires synchronized coordination of subject identity, lip movements, facial expressions, and head poses. The absence of these synchronizations is a fundamental flaw, leading to unrealistic results. To address the critical issue of synchronization, identified as the ''devil'' in creating realistic talking heads, we introduce SyncTalk++, which features a Dynamic Portrait Renderer with Gaussian Splatting to ensure consistent subject identity preservation and a Face-Sync Controller that aligns lip movements with speech while innovatively using a 3D facial blendshape model to reconstruct accurate facial expressions. To ensure natural head movements, we propose a Head-Sync Stabilizer, which optimizes head poses for greater stability. Additionally, SyncTalk++ enhances robustness to out-of-distribution (OOD) audio by incorporating an Expression Generator and a Torso Restorer, which generate speech-matched facial expressions and seamless torso regions. Our approach maintains consistency and continuity in visual details across frames and significantly improves rendering speed and quality, achieving up to 101 frames per second. Extensive experiments and user studies demonstrate that SyncTalk++ outperforms state-of-the-art methods in synchronization and realism. We recommend watching the supplementary video: https://ziqiaopeng.github.io/synctalk++.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14642v1">3DGS-IEval-15K: A Large-scale Image Quality Evaluation Database for 3D Gaussian-Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising approach for novel view synthesis, offering real-time rendering with high visual fidelity. However, its substantial storage requirements present significant challenges for practical applications. While recent state-of-the-art (SOTA) 3DGS methods increasingly incorporate dedicated compression modules, there is a lack of a comprehensive framework to evaluate their perceptual impact. Therefore we present 3DGS-IEval-15K, the first large-scale image quality assessment (IQA) dataset specifically designed for compressed 3DGS representations. Our dataset encompasses 15,200 images rendered from 10 real-world scenes through 6 representative 3DGS algorithms at 20 strategically selected viewpoints, with different compression levels leading to various distortion effects. Through controlled subjective experiments, we collect human perception data from 60 viewers. We validate dataset quality through scene diversity and MOS distribution analysis, and establish a comprehensive benchmark with 30 representative IQA metrics covering diverse types. As the largest-scale 3DGS quality assessment dataset to date, our work provides a foundation for developing 3DGS specialized IQA metrics, and offers essential data for investigating view-dependent quality distribution patterns unique to 3DGS. The database is publicly available at https://github.com/YukeXing/3DGS-IEval-15K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18682v2">Hardware-Rasterized Ray-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      We present a novel, hardware rasterized rendering approach for ray-based 3D Gaussian Splatting (RayGS), obtaining both fast and high-quality results for novel view synthesis. Our work contains a mathematically rigorous and geometrically intuitive derivation about how to efficiently estimate all relevant quantities for rendering RayGS models, structured with respect to standard hardware rasterization shaders. Our solution is the first enabling rendering RayGS models at sufficiently high frame rates to support quality-sensitive applications like Virtual and Mixed Reality. Our second contribution enables alias-free rendering for RayGS, by addressing MIP-related issues arising when rendering diverging scales during training and testing. We demonstrate significant performance gains, across different benchmark scenes, while retaining state-of-the-art appearance quality of RayGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14856v1">Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 9 pages, 3 figures in the main text. Under review for NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Some perspectives naturally provide more information than others. How can an AI system determine which viewpoint offers the most valuable insight for accurate and efficient 3D object reconstruction? Active view selection (AVS) for 3D reconstruction remains a fundamental challenge in computer vision. The aim is to identify the minimal set of views that yields the most accurate 3D reconstruction. Instead of learning radiance fields, like NeRF or 3D Gaussian Splatting, from a current observation and computing uncertainty for each candidate viewpoint, we introduce a novel AVS approach guided by neural uncertainty maps predicted by a lightweight feedforward deep neural network, named UPNet. UPNet takes a single input image of a 3D object and outputs a predicted uncertainty map, representing uncertainty values across all possible candidate viewpoints. By leveraging heuristics derived from observing many natural objects and their associated uncertainty patterns, we train UPNet to learn a direct mapping from viewpoint appearance to uncertainty in the underlying volumetric representations. Next, our approach aggregates all previously predicted neural uncertainty maps to suppress redundant candidate viewpoints and effectively select the most informative one. Using these selected viewpoints, we train 3D neural rendering models and evaluate the quality of novel view synthesis against other competitive AVS methods. Remarkably, despite using half of the viewpoints than the upper bound, our method achieves comparable reconstruction accuracy. In addition, it significantly reduces computational overhead during AVS, achieving up to a 400 times speedup along with over 50\% reductions in CPU, RAM, and GPU usage compared to baseline methods. Notably, our approach generalizes effectively to AVS tasks involving novel object categories, without requiring any additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14229v1">HRGS: Hierarchical Gaussian Splatting for Memory-Efficient High-Resolution 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in real-time 3D scene reconstruction, but faces memory scalability issues in high-resolution scenarios. To address this, we propose Hierarchical Gaussian Splatting (HRGS), a memory-efficient framework with hierarchical block-level optimization. First, we generate a global, coarse Gaussian representation from low-resolution data. Then, we partition the scene into multiple blocks, refining each block with high-resolution data. The partitioning involves two steps: Gaussian partitioning, where irregular scenes are normalized into a bounded cubic space with a uniform grid for task distribution, and training data partitioning, where only relevant observations are retained for each block. By guiding block refinement with the coarse Gaussian prior, we ensure seamless Gaussian fusion across adjacent blocks. To reduce computational demands, we introduce Importance-Driven Gaussian Pruning (IDGP), which computes importance scores for each Gaussian and removes those with minimal contribution, speeding up convergence and reducing memory usage. Additionally, we incorporate normal priors from a pretrained model to enhance surface reconstruction quality. Our method enables high-quality, high-resolution 3D scene reconstruction even under memory constraints. Extensive experiments on three benchmarks show that HRGS achieves state-of-the-art performance in high-resolution novel view synthesis (NVS) and surface reconstruction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14135v1">GAF: Gaussian Action Field as a Dvnamic World Model for Robotic Mlanipulation</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 http://chaiying1.github.io/GAF.github.io/project_page/
    </div>
    <details class="paper-abstract">
      Accurate action inference is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we propose a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing simultaneous modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF supports three key query types: reconstruction of the current scene, prediction of future frames, and estimation of initial action via robot motion. Furthermore, the high-quality current and future frames generated by GAF facilitate manipulation action refinement through a GAF-guided diffusion model. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average success rate in robotic manipulation tasks by 10.33% over state-of-the-art methods. Project page: http://chaiying1.github.io/GAF.github.io/project_page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12727v2">Efficient multi-view training for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a preferred choice alongside Neural Radiance Fields (NeRF) in inverse rendering due to its superior rendering speed. Currently, the common approach in 3DGS is to utilize "single-view" mini-batch training, where only one image is processed per iteration, in contrast to NeRF's "multi-view" mini-batch training, which leverages multiple images. We observe that such single-view training can lead to suboptimal optimization due to increased variance in mini-batch stochastic gradients, highlighting the necessity for multi-view training. However, implementing multi-view training in 3DGS poses challenges. Simply rendering multiple images per iteration incurs considerable overhead and may result in suboptimal Gaussian densification due to its reliance on single-view assumptions. To address these issues, we modify the rasterization process to minimize the overhead associated with multi-view training and propose a 3D distance-aware D-SSIM loss and multi-view adaptive density control that better suits multi-view scenarios. Our experiments demonstrate that the proposed methods significantly enhance the performance of 3DGS and its variants, freeing 3DGS from the constraints of single-view training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13766v1">PF-LHM: 3D Animatable Avatar Reconstruction from Pose-free Articulated Human Images</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Reconstructing an animatable 3D human from casually captured images of an articulated subject without camera or human pose information is a practical yet challenging task due to view misalignment, occlusions, and the absence of structural priors. While optimization-based methods can produce high-fidelity results from monocular or multi-view videos, they require accurate pose estimation and slow iterative optimization, limiting scalability in unconstrained scenarios. Recent feed-forward approaches enable efficient single-image reconstruction but struggle to effectively leverage multiple input images to reduce ambiguity and improve reconstruction accuracy. To address these challenges, we propose PF-LHM, a large human reconstruction model that generates high-quality 3D avatars in seconds from one or multiple casually captured pose-free images. Our approach introduces an efficient Encoder-Decoder Point-Image Transformer architecture, which fuses hierarchical geometric point features and multi-view image features through multimodal attention. The fused features are decoded to recover detailed geometry and appearance, represented using 3D Gaussian splats. Extensive experiments on both real and synthetic datasets demonstrate that our method unifies single- and multi-image 3D human reconstruction, achieving high-fidelity and animatable 3D human avatars without requiring camera and human pose annotations. Code and models will be released to the public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13516v1">Micro-macro Gaussian Splatting with Enhanced Scalability for Unconstrained Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from unconstrained image collections poses significant challenges due to variations in appearance. In this paper, we propose Scalable Micro-macro Wavelet-based Gaussian Splatting (SMW-GS), a novel method that enhances 3D reconstruction across diverse scales by decomposing scene representations into global, refined, and intrinsic components. SMW-GS incorporates the following innovations: Micro-macro Projection, which enables Gaussian points to sample multi-scale details with improved diversity; and Wavelet-based Sampling, which refines feature representations using frequency-domain information to better capture complex scene appearances. To achieve scalability, we further propose a large-scale scene promotion strategy, which optimally assigns camera views to scene partitions by maximizing their contributions to Gaussian points, achieving consistent and high-quality reconstructions even in expansive environments. Extensive experiments demonstrate that SMW-GS significantly outperforms existing methods in both reconstruction quality and scalability, particularly excelling in large-scale urban environments with challenging illumination variations. Project is available at https://github.com/Kidleyh/SMW-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13508v1">Multiview Geometric Regularization of Gaussian Splatting for Accurate Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted to Computer Graphics Forum (EGSR 2025)
    </div>
    <details class="paper-abstract">
      Recent methods, such as 2D Gaussian Splatting and Gaussian Opacity Fields, have aimed to address the geometric inaccuracies of 3D Gaussian Splatting while retaining its superior rendering quality. However, these approaches still struggle to reconstruct smooth and reliable geometry, particularly in scenes with significant color variation across viewpoints, due to their per-point appearance modeling and single-view optimization constraints. In this paper, we propose an effective multiview geometric regularization strategy that integrates multiview stereo (MVS) depth, RGB, and normal constraints into Gaussian Splatting initialization and optimization. Our key insight is the complementary relationship between MVS-derived depth points and Gaussian Splatting-optimized positions: MVS robustly estimates geometry in regions of high color variation through local patch-based matching and epipolar constraints, whereas Gaussian Splatting provides more reliable and less noisy depth estimates near object boundaries and regions with lower color variation. To leverage this insight, we introduce a median depth-based multiview relative depth loss with uncertainty estimation, effectively integrating MVS depth information into Gaussian Splatting optimization. We also propose an MVS-guided Gaussian Splatting initialization to avoid Gaussians falling into suboptimal positions. Extensive experiments validate that our approach successfully combines these strengths, enhancing both geometric accuracy and rendering quality across diverse indoor and outdoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13348v1">TextureSplat: Per-Primitive Texture Mapping for Reflective Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Code will be available at https://github.com/maeyounes/TextureSplat
    </div>
    <details class="paper-abstract">
      Gaussian Splatting have demonstrated remarkable novel view synthesis performance at high rendering frame rates. Optimization-based inverse rendering within complex capture scenarios remains however a challenging problem. A particular case is modelling complex surface light interactions for highly reflective scenes, which results in intricate high frequency specular radiance components. We hypothesize that such challenging settings can benefit from increased representation power. We hence propose a method that tackles this issue through a geometrically and physically grounded Gaussian Splatting borne radiance field, where normals and material properties are spatially variable in the primitive's local space. Using per-primitive texture maps for this purpose, we also propose to harness the GPU hardware to accelerate rendering at test time via unified material texture atlas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21955v2">ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted to IEEE RA-L. Code: https://github.com/Li-Yuetao/ActiveSplat, Project: https://li-yuetao.github.io/ActiveSplat/
    </div>
    <details class="paper-abstract">
      We propose ActiveSplat, an autonomous high-fidelity reconstruction system leveraging Gaussian splatting. Taking advantage of efficient and realistic rendering, the system establishes a unified framework for online mapping, viewpoint selection, and path planning. The key to ActiveSplat is a hybrid map representation that integrates both dense information about the environment and a sparse abstraction of the workspace. Therefore, the system leverages sparse topology for efficient viewpoint sampling and path planning, while exploiting view-dependent dense prediction for viewpoint selection, facilitating efficient decision-making with promising accuracy and completeness. A hierarchical planning strategy based on the topological map is adopted to mitigate repetitive trajectories and improve local granularity given limited time budgets, ensuring high-fidelity reconstruction with photorealistic view synthesis. Extensive experiments and ablation studies validate the efficacy of the proposed method in terms of reconstruction accuracy, data coverage, and exploration efficiency. The released code will be available on our project page: https://li-yuetao.github.io/ActiveSplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13110v1">GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-16
      | 💬 Accepted by CVPR2025
    </div>
    <details class="paper-abstract">
      3D modeling of highly reflective objects remains challenging due to strong view-dependent appearances. While previous SDF-based methods can recover high-quality meshes, they are often time-consuming and tend to produce over-smoothed surfaces. In contrast, 3D Gaussian Splatting (3DGS) offers the advantage of high speed and detailed real-time rendering, but extracting surfaces from the Gaussians can be noisy due to the lack of geometric constraints. To bridge the gap between these approaches, we propose a novel reconstruction method called GS-2DGS for reflective objects based on 2D Gaussian Splatting (2DGS). Our approach combines the rapid rendering capabilities of Gaussian Splatting with additional geometric information from foundation models. Experimental results on synthetic and real datasets demonstrate that our method significantly outperforms Gaussian-based techniques in terms of reconstruction and relighting and achieves performance comparable to SDF-based methods while being an order of magnitude faster. Code is available at https://github.com/hirotong/GS2DGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13196v2">GS-QA: Comprehensive Quality Assessment Benchmark for Gaussian Splatting View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) offers a promising alternative to Neural Radiance Fields (NeRF) for real-time 3D scene rendering. Using a set of 3D Gaussians to represent complex geometry and appearance, GS achieves faster rendering times and reduced memory consumption compared to the neural network approach used in NeRF. However, quality assessment of GS-generated static content is not yet explored in-depth. This paper describes a subjective quality assessment study that aims to evaluate synthesized videos obtained with several static GS state-of-the-art methods. The methods were applied to diverse visual scenes, covering both 360-degree and forward-facing (FF) camera trajectories. Moreover, the performance of 18 objective quality metrics was analyzed using the scores resulting from the subjective study, providing insights into their strengths, limitations, and alignment with human perception. All videos and scores are made available providing a comprehensive database that can be used as benchmark on GS view synthesis and objective quality metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14009v1">GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-06-16
    </div>
    <details class="paper-abstract">
      Autonomous drones capable of interpreting and executing high-level language instructions in unstructured environments remain a long-standing goal. Yet existing approaches are constrained by their dependence on hand-crafted skills, extensive parameter tuning, or computationally intensive models unsuitable for onboard use. We introduce GRaD-Nav++, a lightweight Vision-Language-Action (VLA) framework that runs fully onboard and follows natural-language commands in real time. Our policy is trained in a photorealistic 3D Gaussian Splatting (3DGS) simulator via Differentiable Reinforcement Learning (DiffRL), enabling efficient learning of low-level control from visual and linguistic inputs. At its core is a Mixture-of-Experts (MoE) action head, which adaptively routes computation to improve generalization while mitigating forgetting. In multi-task generalization experiments, GRaD-Nav++ achieves a success rate of 83% on trained tasks and 75% on unseen tasks in simulation. When deployed on real hardware, it attains 67% success on trained tasks and 50% on unseen ones. In multi-environment adaptation experiments, GRaD-Nav++ achieves an average success rate of 81% across diverse simulated environments and 67% across varied real-world settings. These results establish a new benchmark for fully onboard Vision-Language-Action (VLA) flight and demonstrate that compact, efficient models can enable reliable, language-guided navigation without relying on external infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12945v1">Metropolis-Hastings Sampling for 3D Gaussian Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 Project Page: https://hjhyunjinkim.github.io/MH-3DGS
    </div>
    <details class="paper-abstract">
      We propose an adaptive sampling framework for 3D Gaussian Splatting (3DGS) that leverages comprehensive multi-view photometric error signals within a unified Metropolis-Hastings approach. Traditional 3DGS methods heavily rely on heuristic-based density-control mechanisms (e.g., cloning, splitting, and pruning), which can lead to redundant computations or the premature removal of beneficial Gaussians. Our framework overcomes these limitations by reformulating densification and pruning as a probabilistic sampling process, dynamically inserting and relocating Gaussians based on aggregated multi-view errors and opacity scores. Guided by Bayesian acceptance tests derived from these error-based importance scores, our method substantially reduces reliance on heuristics, offers greater flexibility, and adaptively infers Gaussian distributions without requiring predefined scene complexity. Experiments on benchmark datasets, including Mip-NeRF360, Tanks and Temples, and Deep Blending, show that our approach reduces the number of Gaussians needed, enhancing computational efficiency while matching or modestly surpassing the view-synthesis quality of state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13335v3">Deblur-Avatar: Animatable Avatars from Motion-Blurred Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for modeling high-fidelity, animatable 3D human avatars from motion-blurred monocular video inputs. Motion blur is prevalent in real-world dynamic video capture, especially due to human movements in 3D human avatar modeling. Existing methods either (1) assume sharp image inputs, failing to address the detail loss introduced by motion blur, or (2) mainly consider blur by camera movements, neglecting the human motion blur which is more common in animatable avatars. Our proposed approach integrates a human movement-based motion blur model into 3D Gaussian Splatting (3DGS). By explicitly modeling human motion trajectories during exposure time, we jointly optimize the trajectories and 3D Gaussians to reconstruct sharp, high-quality human avatars. We employ a pose-dependent fusion mechanism to distinguish moving body regions, optimizing both blurred and sharp areas effectively. Extensive experiments on synthetic and real-world datasets demonstrate that our method significantly outperforms existing methods in rendering quality and quantitative metrics, producing sharp avatar reconstructions and enabling real-time rendering under challenging motion blur conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12787v1">Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      Modeling the wireless radiance field (WRF) is fundamental to modern communication systems, enabling key tasks such as localization, sensing, and channel estimation. Traditional approaches, which rely on empirical formulas or physical simulations, often suffer from limited accuracy or require strong scene priors. Recent neural radiance field (NeRF-based) methods improve reconstruction fidelity through differentiable volumetric rendering, but their reliance on computationally expensive multilayer perceptron (MLP) queries hinders real-time deployment. To overcome these challenges, we introduce Gaussian splatting (GS) to the wireless domain, leveraging its efficiency in modeling optical radiance fields to enable compact and accurate WRF reconstruction. Specifically, we propose SwiftWRF, a deformable 2D Gaussian splatting framework that synthesizes WRF spectra at arbitrary positions under single-sided transceiver mobility. SwiftWRF employs CUDA-accelerated rasterization to render spectra at over 100000 fps and uses a lightweight MLP to model the deformation of 2D Gaussians, effectively capturing mobility-induced WRF variations. In addition to novel spectrum synthesis, the efficacy of SwiftWRF is further underscored in its applications in angle-of-arrival (AoA) and received signal strength indicator (RSSI) prediction. Experiments conducted on both real-world and synthetic indoor scenes demonstrate that SwiftWRF can reconstruct WRF spectra up to 500x faster than existing state-of-the-art methods, while significantly enhancing its signal quality. Code and datasets will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12727v1">Efficient multi-view training for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-15
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a preferred choice alongside Neural Radiance Fields (NeRF) in inverse rendering due to its superior rendering speed. Currently, the common approach in 3DGS is to utilize "single-view" mini-batch training, where only one image is processed per iteration, in contrast to NeRF's "multi-view" mini-batch training, which leverages multiple images. We observe that such single-view training can lead to suboptimal optimization due to increased variance in mini-batch stochastic gradients, highlighting the necessity for multi-view training. However, implementing multi-view training in 3DGS poses challenges. Simply rendering multiple images per iteration incurs considerable overhead and may result in suboptimal Gaussian densification due to its reliance on single-view assumptions. To address these issues, we modify the rasterization process to minimize the overhead associated with multi-view training and propose a 3D distance-aware D-SSIM loss and multi-view adaptive density control that better suits multi-view scenarios. Our experiments demonstrate that the proposed methods significantly enhance the performance of 3DGS and its variants, freeing 3DGS from the constraints of single-view training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12716v1">Generative 4D Scene Gaussian Splatting with Object View-Synthesis Priors</a></div>
    <div class="paper-meta">
      📅 2025-06-15
      | 💬 This is an updated and extended version of our CVPR paper "Robust Multi-Object 4D Generation in Complex Video Scenarios"
    </div>
    <details class="paper-abstract">
      We tackle the challenge of generating dynamic 4D scenes from monocular, multi-object videos with heavy occlusions, and introduce GenMOJO, a novel approach that integrates rendering-based deformable 3D Gaussian optimization with generative priors for view synthesis. While existing models perform well on novel view synthesis for isolated objects, they struggle to generalize to complex, cluttered scenes. To address this, GenMOJO decomposes the scene into individual objects, optimizing a differentiable set of deformable Gaussians per object. This object-wise decomposition allows leveraging object-centric diffusion models to infer unobserved regions in novel viewpoints. It performs joint Gaussian splatting to render the full scene, capturing cross-object occlusions, and enabling occlusion-aware supervision. To bridge the gap between object-centric priors and the global frame-centric coordinate system of videos, GenMOJO uses differentiable transformations that align generative and rendering constraints within a unified framework. The resulting model generates 4D object reconstructions over space and time, and produces accurate 2D and 3D point tracks from monocular input. Quantitative evaluations and perceptual human studies confirm that GenMOJO generates more realistic novel views of scenes and produces more accurate point tracks compared to existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20522v3">MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 CVPR 2025; Project page:https://maskgaussian.github.io/
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in novel view synthesis and real-time rendering, the high memory consumption due to the use of millions of Gaussians limits its practicality. To mitigate this issue, improvements have been made by pruning unnecessary Gaussians, either through a hand-crafted criterion or by using learned masks. However, these methods deterministically remove Gaussians based on a snapshot of the pruning moment, leading to sub-optimized reconstruction performance from a long-term perspective. To address this issue, we introduce MaskGaussian, which models Gaussians as probabilistic entities rather than permanently removing them, and utilize them according to their probability of existence. To achieve this, we propose a masked-rasterization technique that enables unused yet probabilistically existing Gaussians to receive gradients, allowing for dynamic assessment of their contribution to the evolving scene and adjustment of their probability of existence. Hence, the importance of Gaussians iteratively changes and the pruned Gaussians are selected diversely. Extensive experiments demonstrate the superiority of the proposed method in achieving better rendering quality with fewer Gaussians than previous pruning methods, pruning over 60% of Gaussians on average with only a 0.02 PSNR decline. Our code can be found at: https://github.com/kaikai23/MaskGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05511v4">Free Your Hands: Lightweight Turntable-Based Object Capture Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Siggraph Asia Submission
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) from multiple captured photos of an object is a widely studied problem. Achieving high quality typically requires dense sampling of input views, which can lead to frustrating manual labor. Manually positioning cameras to maintain an optimal desired distribution can be difficult for humans, and if a good distribution is found, it is not easy to replicate. Additionally, the captured data can suffer from motion blur and defocus due to human error. In this paper, we use a lightweight object capture pipeline to reduce the manual workload and standardize the acquisition setup, with a consumer turntable to carry the object and a tripod to hold the camera. Of course, turntables and gantry systems have been frequently used to automatically capture dense samples under various views and lighting conditions; the key difference is that we use a turntable under natural environment lighting. This way, we can easily capture hundreds of valid images in several minutes without hands-on effort. However, in the object reference frame, the light conditions vary (rotate); this does not match the assumptions of standard NVS methods like 3D Gaussian splatting (3DGS). We design a neural radiance representation conditioned on light rotations, which addresses this issue and allows rendering with novel light rotations as an additional benefit. We further study the behavior of rotations and find optimal capturing strategies. We demonstrate our pipeline using 3DGS as the underlying framework, achieving higher quality and showcasing the method's potential for novel lighting and harmonization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16619v4">Topology-Aware 3D Gaussian Splatting: Leveraging Persistent Homology for Optimized Structural Integrity</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 18 pages, 12 figures, includes appendix. Accepted as oral presentation at AAAI 2025 (Conference on Artificial Intelligence). Official conference version: 10 pages, 6 figures. ISBN (Print): 978-1-57735-897-8. Conference website: https://aaai.org/conference/aaai/aaai-25/
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has emerged as a crucial technique for representing discrete volumetric radiance fields. It leverages unique parametrization to mitigate computational demands in scene optimization. This work introduces Topology-Aware 3D Gaussian Splatting (Topology-GS), which addresses two key limitations in current approaches: compromised pixel-level structural integrity due to incomplete initial geometric coverage, and inadequate feature-level integrity from insufficient topological constraints during optimization. To overcome these limitations, Topology-GS incorporates a novel interpolation strategy, Local Persistent Voronoi Interpolation (LPVI), and a topology-focused regularization term based on persistent barcodes, named PersLoss. LPVI utilizes persistent homology to guide adaptive interpolation, enhancing point coverage in low-curvature areas while preserving topological structure. PersLoss aligns the visual perceptual similarity of rendered images with ground truth by constraining distances between their topological features. Comprehensive experiments on three novel-view synthesis benchmarks demonstrate that Topology-GS outperforms existing methods in terms of PSNR, SSIM, and LPIPS metrics, while maintaining efficient memory usage. This study pioneers the integration of topology with 3D-GS, laying the groundwork for future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12400v1">Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-14
      | 💬 Accepted to International Conference on Machine Learning (ICML) 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis. However, existing methods struggle to adaptively optimize the distribution of Gaussian primitives based on scene characteristics, making it challenging to balance reconstruction quality and efficiency. Inspired by human perception, we propose scene-adaptive perceptual densification for Gaussian Splatting (Perceptual-GS), a novel framework that integrates perceptual sensitivity into the 3DGS training process to address this challenge. We first introduce a perception-aware representation that models human visual sensitivity while constraining the number of Gaussian primitives. Building on this foundation, we develop a \cameraready{perceptual sensitivity-adaptive distribution} to allocate finer Gaussian granularity to visually critical regions, enhancing reconstruction quality and robustness. Extensive evaluations on multiple datasets, including BungeeNeRF for large-scale scenes, demonstrate that Perceptual-GS achieves state-of-the-art performance in reconstruction quality, efficiency, and robustness. The code is publicly available at: https://github.com/eezkni/Perceptual-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12184v1">SPLATART: Articulated Gaussian Splatting with Estimated Object Structure</a></div>
    <div class="paper-meta">
      📅 2025-06-13
      | 💬 7 pages, Accepted to the 2025 RSS Workshop on Gaussian Representations for Robot Autonomy. Contact: Stanley Lewis, stanlew@umich.edu
    </div>
    <details class="paper-abstract">
      Representing articulated objects remains a difficult problem within the field of robotics. Objects such as pliers, clamps, or cabinets require representations that capture not only geometry and color information, but also part seperation, connectivity, and joint parametrization. Furthermore, learning these representations becomes even more difficult with each additional degree of freedom. Complex articulated objects such as robot arms may have seven or more degrees of freedom, and the depth of their kinematic tree may be notably greater than the tools, drawers, and cabinets that are the typical subjects of articulated object research. To address these concerns, we introduce SPLATART - a pipeline for learning Gaussian splat representations of articulated objects from posed images, of which a subset contains image space part segmentations. SPLATART disentangles the part separation task from the articulation estimation task, allowing for post-facto determination of joint estimation and representation of articulated objects with deeper kinematic trees than previously exhibited. In this work, we present data on the SPLATART pipeline as applied to the syntheic Paris dataset objects, and qualitative results on a real-world object under spare segmentation supervision. We additionally present on articulated serial chain manipulators to demonstrate usage on deeper kinematic tree structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09417v2">ODG: Occupancy Prediction Using Dual Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Occupancy prediction infers fine-grained 3D geometry and semantics from camera images of the surrounding environment, making it a critical perception task for autonomous driving. Existing methods either adopt dense grids as scene representation, which is difficult to scale to high resolution, or learn the entire scene using a single set of sparse queries, which is insufficient to handle the various object characteristics. In this paper, we present ODG, a hierarchical dual sparse Gaussian representation to effectively capture complex scene dynamics. Building upon the observation that driving scenes can be universally decomposed into static and dynamic counterparts, we define dual Gaussian queries to better model the diverse scene objects. We utilize a hierarchical Gaussian transformer to predict the occupied voxel centers and semantic classes along with the Gaussian parameters. Leveraging the real-time rendering capability of 3D Gaussian Splatting, we also impose rendering supervision with available depth and semantic map annotations injecting pixel-level alignment to boost occupancy learning. Extensive experiments on the Occ3D-nuScenes and Occ3D-Waymo benchmarks demonstrate our proposed method sets new state-of-the-art results while maintaining low inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10335v1">PointGS: Point Attention-Aware Sparse View Synthesis with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) is an innovative rendering technique that surpasses the neural radiance field (NeRF) in both rendering speed and visual quality by leveraging an explicit 3D scene representation. Existing 3DGS approaches require a large number of calibrated views to generate a consistent and complete scene representation. When input views are limited, 3DGS tends to overfit the training views, leading to noticeable degradation in rendering quality. To address this limitation, we propose a Point-wise Feature-Aware Gaussian Splatting framework that enables real-time, high-quality rendering from sparse training views. Specifically, we first employ the latest stereo foundation model to estimate accurate camera poses and reconstruct a dense point cloud for Gaussian initialization. We then encode the colour attributes of each 3D Gaussian by sampling and aggregating multiscale 2D appearance features from sparse inputs. To enhance point-wise appearance representation, we design a point interaction network based on a self-attention mechanism, allowing each Gaussian point to interact with its nearest neighbors. These enriched features are subsequently decoded into Gaussian parameters through two lightweight multi-layer perceptrons (MLPs) for final rendering. Extensive experiments on diverse benchmarks demonstrate that our method significantly outperforms NeRF-based approaches and achieves competitive performance under few-shot settings compared to the state-of-the-art 3DGS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12836v5">CompMarkGS: Robust Watermarking for Compressed 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 28 pages, 19 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly adopted in various academic and commercial applications due to its real-time and high-quality rendering capabilities, emphasizing the growing need for copyright protection technologies for 3DGS. However, the large model size of 3DGS requires developing efficient compression techniques. This highlights the necessity of an integrated framework that addresses copyright protection and data compression for 3D content. Nevertheless, existing 3DGS watermarking methods significantly degrade watermark performance under 3DGS compression methods, particularly quantization-based approaches that achieve superior compression performance. To ensure reliable watermark detection under compression, we propose a compression-tolerant anchor-based 3DGS watermarking, which preserves watermark integrity and rendering quality. This is achieved by introducing anchor-based 3DGS watermarking. We embed the watermark into the anchor attributes, particularly the anchor feature, to enhance security and rendering quality. We also propose a quantization distortion layer that injects quantization noise during training, preserving the watermark after quantization-based compression. Moreover, we employ a frequency-aware anchor growing strategy that improves rendering quality and watermark performance by effectively identifying Gaussians in high-frequency regions. Extensive experiments demonstrate that our proposed method preserves the watermark even under compression and maintains high rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11252v1">Anti-Aliased 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Code will be available at https://github.com/maeyounes/AA-2DGS
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting (2DGS) has recently emerged as a promising method for novel view synthesis and surface reconstruction, offering better view-consistency and geometric accuracy than volumetric 3DGS. However, 2DGS suffers from severe aliasing artifacts when rendering at different sampling rates than those used during training, limiting its practical applications in scenarios requiring camera zoom or varying fields of view. We identify that these artifacts stem from two key limitations: the lack of frequency constraints in the representation and an ineffective screen-space clamping approach. To address these issues, we present AA-2DGS, an antialiased formulation of 2D Gaussian Splatting that maintains its geometric benefits while significantly enhancing rendering quality across different scales. Our method introduces a world space flat smoothing kernel that constrains the frequency content of 2D Gaussian primitives based on the maximal sampling frequency from training views, effectively eliminating high-frequency artifacts when zooming in. Additionally, we derive a novel object space Mip filter by leveraging an affine approximation of the ray-splat intersection mapping, which allows us to efficiently apply proper anti-aliasing directly in the local space of each splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09997v1">DGS-LRM: Real-Time Deformable 3D Gaussian Reconstruction From Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Project page: https://hubert0527.github.io/dgslrm/
    </div>
    <details class="paper-abstract">
      We introduce the Deformable Gaussian Splats Large Reconstruction Model (DGS-LRM), the first feed-forward method predicting deformable 3D Gaussian splats from a monocular posed video of any dynamic scene. Feed-forward scene reconstruction has gained significant attention for its ability to rapidly create digital replicas of real-world environments. However, most existing models are limited to static scenes and fail to reconstruct the motion of moving objects. Developing a feed-forward model for dynamic scene reconstruction poses significant challenges, including the scarcity of training data and the need for appropriate 3D representations and training paradigms. To address these challenges, we introduce several key technical contributions: an enhanced large-scale synthetic dataset with ground-truth multi-view videos and dense 3D scene flow supervision; a per-pixel deformable 3D Gaussian representation that is easy to learn, supports high-quality dynamic view synthesis, and enables long-range 3D tracking; and a large transformer network that achieves real-time, generalizable dynamic scene reconstruction. Extensive qualitative and quantitative experiments demonstrate that DGS-LRM achieves dynamic scene reconstruction quality comparable to optimization-based methods, while significantly outperforming the state-of-the-art predictive dynamic reconstruction method on real-world examples. Its predicted physically grounded 3D deformation is accurate and can readily adapt for long-range 3D tracking tasks, achieving performance on par with state-of-the-art monocular video 3D tracking methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09952v1">UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted to CVPR 2025
    </div>
    <details class="paper-abstract">
      The scale diversity of point cloud data presents significant challenges in developing unified representation learning techniques for 3D vision. Currently, there are few unified 3D models, and no existing pre-training method is equally effective for both object- and scene-level point clouds. In this paper, we introduce UniPre3D, the first unified pre-training method that can be seamlessly applied to point clouds of any scale and 3D models of any architecture. Our approach predicts Gaussian primitives as the pre-training task and employs differentiable Gaussian splatting to render images, enabling precise pixel-level supervision and end-to-end optimization. To further regulate the complexity of the pre-training task and direct the model's focus toward geometric structures, we integrate 2D features from pre-trained image models to incorporate well-established texture knowledge. We validate the universal effectiveness of our proposed method through extensive experiments across a variety of object- and scene-level tasks, using diverse point cloud models as backbones. Code is available at https://github.com/wangzy22/UniPre3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09836v1">DynaSplat: Dynamic-Static Gaussian Splatting with Hierarchical Motion Decomposition for Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Reconstructing intricate, ever-changing environments remains a central ambition in computer vision, yet existing solutions often crumble before the complexity of real-world dynamics. We present DynaSplat, an approach that extends Gaussian Splatting to dynamic scenes by integrating dynamic-static separation and hierarchical motion modeling. First, we classify scene elements as static or dynamic through a novel fusion of deformation offset statistics and 2D motion flow consistency, refining our spatial representation to focus precisely where motion matters. We then introduce a hierarchical motion modeling strategy that captures both coarse global transformations and fine-grained local movements, enabling accurate handling of intricate, non-rigid motions. Finally, we integrate physically-based opacity estimation to ensure visually coherent reconstructions, even under challenging occlusions and perspective shifts. Extensive experiments on challenging datasets reveal that DynaSplat not only surpasses state-of-the-art alternatives in accuracy and realism but also provides a more intuitive, compact, and efficient route to dynamic scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13610v3">Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Existing approaches to drone visual geo-localization predominantly adopt the image-based setting, where a single drone-view snapshot is matched with images from other platforms. Such task formulation, however, underutilizes the inherent video output of the drone and is sensitive to occlusions and viewpoint disparity. To address these limitations, we formulate a new video-based drone geo-localization task and propose the Video2BEV paradigm. This paradigm transforms the video into a Bird's Eye View (BEV), simplifying the subsequent \textbf{inter-platform} matching process. In particular, we employ Gaussian Splatting to reconstruct a 3D scene and obtain the BEV projection. Different from the existing transform methods, \eg, polar transform, our BEVs preserve more fine-grained details without significant distortion. To facilitate the discriminative \textbf{intra-platform} representation learning, our Video2BEV paradigm also incorporates a diffusion-based module for generating hard negative samples. To validate our approach, we introduce UniV, a new video-based geo-localization dataset that extends the image-based University-1652 dataset. UniV features flight paths at $30^\circ$ and $45^\circ$ elevation angles with increased frame rates of up to 10 frames per second (FPS). Extensive experiments on the UniV dataset show that our Video2BEV paradigm achieves competitive recall rates and outperforms conventional video-based methods. Compared to other competitive methods, our proposed approach exhibits robustness at lower elevations with more occlusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09663v1">Self-Supervised Multi-Part Articulated Objects Modeling via Deformable Gaussian Splatting and Progressive Primitive Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Articulated objects are ubiquitous in everyday life, and accurate 3D representations of their geometry and motion are critical for numerous applications. However, in the absence of human annotation, existing approaches still struggle to build a unified representation for objects that contain multiple movable parts. We introduce DeGSS, a unified framework that encodes articulated objects as deformable 3D Gaussian fields, embedding geometry, appearance, and motion in one compact representation. Each interaction state is modeled as a smooth deformation of a shared field, and the resulting deformation trajectories guide a progressive coarse-to-fine part segmentation that identifies distinct rigid components, all in an unsupervised manner. The refined field provides a spatially continuous, fully decoupled description of every part, supporting part-level reconstruction and precise modeling of their kinematic relationships. To evaluate generalization and realism, we enlarge the synthetic PartNet-Mobility benchmark and release RS-Art, a real-to-sim dataset that pairs RGB captures with accurately reverse-engineered 3D models. Extensive experiments demonstrate that our method outperforms existing methods in both accuracy and stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13639v2">4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp
    </div>
    <details class="paper-abstract">
      4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly being used for odometry and SLAM applications. However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing point cloud matching based solutions, especially those originally intended for more accurate sensors such as LiDAR. Inspired by visual odometry research around 3D Gaussian Splatting, in this paper we propose using freely positioned 3D Gaussians to create a summarized representation of a radar point cloud tolerant to sensor noise, and subsequently leverage its inherent probability distribution function for registration (similar to NDT). Moreover, we propose simultaneously optimizing multiple scan matching hypotheses in order to further increase the robustness of the system against local optima of the function. Finally, we fuse our Gaussian modeling and scan matching algorithms into an EKF radar-inertial odometry system designed after current best practices. Experiments using publicly available 4D radar datasets show that our Gaussian-based odometry is comparable to existing registration algorithms, outperforming them in several sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09534v1">Gaussian Herding across Pens: An Optimal Transport Perspective on Global Gaussian Reduction for 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 18 pages, 8 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful technique for radiance field rendering, but it typically requires millions of redundant Gaussian primitives, overwhelming memory and rendering budgets. Existing compaction approaches address this by pruning Gaussians based on heuristic importance scores, without global fidelity guarantee. To bridge this gap, we propose a novel optimal transport perspective that casts 3DGS compaction as global Gaussian mixture reduction. Specifically, we first minimize the composite transport divergence over a KD-tree partition to produce a compact geometric representation, and then decouple appearance from geometry by fine-tuning color and opacity attributes with far fewer Gaussian primitives. Experiments on benchmark datasets show that our method (i) yields negligible loss in rendering quality (PSNR, SSIM, LPIPS) compared to vanilla 3DGS with only 10% Gaussians; and (ii) consistently outperforms state-of-the-art 3DGS compaction techniques. Notably, our method is applicable to any stage of vanilla or accelerated 3DGS pipelines, providing an efficient and agnostic pathway to lightweight neural rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09518v1">HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from monocular videos remains a fundamental challenge in 3D vision. While 3D Gaussian Splatting (3DGS) achieves real-time rendering in static settings, extending it to dynamic scenes is challenging due to the difficulty of learning structured and temporally consistent motion representations. This challenge often manifests as three limitations in existing methods: redundant Gaussian updates, insufficient motion supervision, and weak modeling of complex non-rigid deformations. These issues collectively hinder coherent and efficient dynamic reconstruction. To address these limitations, we propose HAIF-GS, a unified framework that enables structured and consistent dynamic modeling through sparse anchor-driven deformation. It first identifies motion-relevant regions via an Anchor Filter to suppresses redundant updates in static areas. A self-supervised Induced Flow-Guided Deformation module induces anchor motion using multi-frame feature aggregation, eliminating the need for explicit flow labels. To further handle fine-grained deformations, a Hierarchical Anchor Propagation mechanism increases anchor resolution based on motion complexity and propagates multi-level transformations. Extensive experiments on synthetic and real-world benchmarks validate that HAIF-GS significantly outperforms prior dynamic 3DGS methods in rendering quality, temporal coherence, and reconstruction efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08777v2">Gaussian2Scene: 3D Scene Representation Learning via Self-supervised Learning with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Self-supervised learning (SSL) for point cloud pre-training has become a cornerstone for many 3D vision tasks, enabling effective learning from large-scale unannotated data. At the scene level, existing SSL methods often incorporate volume rendering into the pre-training framework, using RGB-D images as reconstruction signals to facilitate cross-modal learning. This strategy promotes alignment between 2D and 3D modalities and enables the model to benefit from rich visual cues in the RGB-D inputs. However, these approaches are limited by their reliance on implicit scene representations and high memory demands. Furthermore, since their reconstruction objectives are applied only in 2D space, they often fail to capture underlying 3D geometric structures. To address these challenges, we propose Gaussian2Scene, a novel scene-level SSL framework that leverages the efficiency and explicit nature of 3D Gaussian Splatting (3DGS) for pre-training. The use of 3DGS not only alleviates the computational burden associated with volume rendering but also supports direct 3D scene reconstruction, thereby enhancing the geometric understanding of the backbone network. Our approach follows a progressive two-stage training strategy. In the first stage, a dual-branch masked autoencoder learns both 2D and 3D scene representations. In the second stage, we initialize training with reconstructed point clouds and further supervise learning using the geometric locations of Gaussian primitives and rendered RGB images. This process reinforces both geometric and cross-modal learning. We demonstrate the effectiveness of Gaussian2Scene across several downstream 3D object detection tasks, showing consistent improvements over existing pre-training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09479v1">TinySplat: Feedforward Approach for Generating Compact 3D Scene Representation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      The recent development of feedforward 3D Gaussian Splatting (3DGS) presents a new paradigm to reconstruct 3D scenes. Using neural networks trained on large-scale multi-view datasets, it can directly infer 3DGS representations from sparse input views. Although the feedforward approach achieves high reconstruction speed, it still suffers from the substantial storage cost of 3D Gaussians. Existing 3DGS compression methods relying on scene-wise optimization are not applicable due to architectural incompatibilities. To overcome this limitation, we propose TinySplat, a complete feedforward approach for generating compact 3D scene representations. Built upon standard feedforward 3DGS methods, TinySplat integrates a training-free compression framework that systematically eliminates key sources of redundancy. Specifically, we introduce View-Projection Transformation (VPT) to reduce geometric redundancy by projecting geometric parameters into a more compact space. We further present Visibility-Aware Basis Reduction (VABR), which mitigates perceptual redundancy by aligning feature energy along dominant viewing directions via basis transformation. Lastly, spatial redundancy is addressed through an off-the-shelf video codec. Comprehensive experimental results on multiple benchmark datasets demonstrate that TinySplat achieves over 100x compression for 3D Gaussian data generated by feedforward methods. Compared to the state-of-the-art compression approach, we achieve comparable quality with only 6% of the storage size. Meanwhile, our compression framework requires only 25% of the encoding time and 1% of the decoding time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12894v2">FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Project page: https://3dgs-flod.github.io/flod/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) and its subsequent works are restricted to specific hardware setups, either on only low-cost or on only high-end configurations. Approaches aimed at reducing 3DGS memory usage enable rendering on low-cost GPU but compromise rendering quality, which fails to leverage the hardware capabilities in the case of higher-end GPU. Conversely, methods that enhance rendering quality require high-end GPU with large VRAM, making such methods impractical for lower-end devices with limited memory capacity. Consequently, 3DGS-based works generally assume a single hardware setup and lack the flexibility to adapt to varying hardware constraints. To overcome this limitation, we propose Flexible Level of Detail (FLoD) for 3DGS. FLoD constructs a multi-level 3DGS representation through level-specific 3D scale constraints, where each level independently reconstructs the entire scene with varying detail and GPU memory usage. A level-by-level training strategy is introduced to ensure structural consistency across levels. Furthermore, the multi-level structure of FLoD allows selective rendering of image regions at different detail levels, providing additional memory-efficient rendering options. To our knowledge, among prior works which incorporate the concept of Level of Detail (LoD) with 3DGS, FLoD is the first to follow the core principle of LoD by offering adjustable options for a broad range of GPU settings. Experiments demonstrate that FLoD provides various rendering options with trade-offs between quality and memory usage, enabling real-time rendering under diverse memory constraints. Furthermore, we show that FLoD generalizes to different 3DGS frameworks, indicating its potential for integration into future state-of-the-art developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09378v1">UniForward: Unified 3D Scene and Semantic Field Reconstruction via Feed-Forward Gaussian Splatting from Only Sparse-View Images</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      We propose a feed-forward Gaussian Splatting model that unifies 3D scene and semantic field reconstruction. Combining 3D scenes with semantic fields facilitates the perception and understanding of the surrounding environment. However, key challenges include embedding semantics into 3D representations, achieving generalizable real-time reconstruction, and ensuring practical applicability by using only images as input without camera parameters or ground truth depth. To this end, we propose UniForward, a feed-forward model to predict 3D Gaussians with anisotropic semantic features from only uncalibrated and unposed sparse-view images. To enable the unified representation of the 3D scene and semantic field, we embed semantic features into 3D Gaussians and predict them through a dual-branch decoupled decoder. During training, we propose a loss-guided view sampler to sample views from easy to hard, eliminating the need for ground truth depth or masks required by previous methods and stabilizing the training process. The whole model can be trained end-to-end using a photometric loss and a distillation loss that leverages semantic features from a pre-trained 2D semantic model. At the inference stage, our UniForward can reconstruct 3D scenes and the corresponding semantic fields in real time from only sparse-view images. The reconstructed 3D scenes achieve high-quality rendering, and the reconstructed 3D semantic field enables the rendering of view-consistent semantic features from arbitrary views, which can be further decoded into dense segmentation masks in an open-vocabulary manner. Experiments on novel view synthesis and novel view segmentation demonstrate that our method achieves state-of-the-art performances for unifying 3D scene and semantic field reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08071v2">GigaSLAM: Large-Scale Monocular SLAM with Hierarchical Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2025-06-10
    </div>
    <details class="paper-abstract">
      Tracking and mapping in large-scale, unbounded outdoor environments using only monocular RGB input presents substantial challenges for existing SLAM systems. Traditional Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) SLAM methods are typically limited to small, bounded indoor settings. To overcome these challenges, we introduce GigaSLAM, the first RGB NeRF / 3DGS-based SLAM framework for kilometer-scale outdoor environments, as demonstrated on the KITTI, KITTI 360, 4 Seasons and A2D2 datasets. Our approach employs a hierarchical sparse voxel map representation, where Gaussians are decoded by neural networks at multiple levels of detail. This design enables efficient, scalable mapping and high-fidelity viewpoint rendering across expansive, unbounded scenes. For front-end tracking, GigaSLAM utilizes a metric depth model combined with epipolar geometry and PnP algorithms to accurately estimate poses, while incorporating a Bag-of-Words-based loop closure mechanism to maintain robust alignment over long trajectories. Consequently, GigaSLAM delivers high-precision tracking and visually faithful rendering on urban outdoor benchmarks, establishing a robust SLAM solution for large-scale, long-term scenarios, and significantly extending the applicability of Gaussian Splatting SLAM systems to unbounded outdoor environments. GitHub: https://github.com/DengKaiCQ/GigaSLAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08862v1">StreamSplat: Towards Online Dynamic 3D Reconstruction from Uncalibrated Video Streams</a></div>
    <div class="paper-meta">
      📅 2025-06-10
    </div>
    <details class="paper-abstract">
      Real-time reconstruction of dynamic 3D scenes from uncalibrated video streams is crucial for numerous real-world applications. However, existing methods struggle to jointly address three key challenges: 1) processing uncalibrated inputs in real time, 2) accurately modeling dynamic scene evolution, and 3) maintaining long-term stability and computational efficiency. To this end, we introduce StreamSplat, the first fully feed-forward framework that transforms uncalibrated video streams of arbitrary length into dynamic 3D Gaussian Splatting (3DGS) representations in an online manner, capable of recovering scene dynamics from temporally local observations. We propose two key technical innovations: a probabilistic sampling mechanism in the static encoder for 3DGS position prediction, and a bidirectional deformation field in the dynamic decoder that enables robust and efficient dynamic modeling. Extensive experiments on static and dynamic benchmarks demonstrate that StreamSplat consistently outperforms prior works in both reconstruction quality and dynamic scene modeling, while uniquely supporting online reconstruction of arbitrarily long video streams. Code and models are available at https://github.com/nickwzk/StreamSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08710v1">SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-10
      | 💬 15 pages, codes, data and benchmark will be released
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) serves as a highly performant and efficient encoding of scene geometry, appearance, and semantics. Moreover, grounding language in 3D scenes has proven to be an effective strategy for 3D scene understanding. Current Language Gaussian Splatting line of work fall into three main groups: (i) per-scene optimization-based, (ii) per-scene optimization-free, and (iii) generalizable approach. However, most of them are evaluated only on rendered 2D views of a handful of scenes and viewpoints close to the training views, limiting ability and insight into holistic 3D understanding. To address this gap, we propose the first large-scale benchmark that systematically assesses these three groups of methods directly in 3D space, evaluating on 1060 scenes across three indoor datasets and one outdoor dataset. Benchmark results demonstrate a clear advantage of the generalizable paradigm, particularly in relaxing the scene-specific limitation, enabling fast feed-forward inference on novel scenes, and achieving superior segmentation performance. We further introduce GaussianWorld-49K a carefully curated 3DGS dataset comprising around 49K diverse indoor and outdoor scenes obtained from multiple sources, with which we demonstrate the generalizable approach could harness strong data priors. Our codes, benchmark, and datasets will be made public to accelerate research in generalizable 3DGS scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08704v1">TraGraph-GS: Trajectory Graph-based Gaussian Splatting for Arbitrary Large-Scale Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-06-10
    </div>
    <details class="paper-abstract">
      High-quality novel view synthesis for large-scale scenes presents a challenging dilemma in 3D computer vision. Existing methods typically partition large scenes into multiple regions, reconstruct a 3D representation using Gaussian splatting for each region, and eventually merge them for novel view rendering. They can accurately render specific scenes, yet they do not generalize effectively for two reasons: (1) rigid spatial partition techniques struggle with arbitrary camera trajectories, and (2) the merging of regions results in Gaussian overlap to distort texture details. To address these challenges, we propose TraGraph-GS, leveraging a trajectory graph to enable high-precision rendering for arbitrarily large-scale scenes. We present a spatial partitioning method for large-scale scenes based on graphs, which incorporates a regularization constraint to enhance the rendering of textures and distant objects, as well as a progressive rendering strategy to mitigate artifacts caused by Gaussian overlap. Experimental results demonstrate its superior performance both on four aerial and four ground datasets and highlight its remarkable efficiency: our method achieves an average improvement of 1.86 dB in PSNR on aerial datasets and 1.62 dB on ground datasets compared to state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08350v1">Complex-Valued Holographic Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-06-10
      | 💬 28 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Modeling the full properties of light, including both amplitude and phase, in 3D representations is crucial for advancing physically plausible rendering, particularly in holographic displays. To support these features, we propose a novel representation that optimizes 3D scenes without relying on intensity-based intermediaries. We reformulate 3D Gaussian splatting with complex-valued Gaussian primitives, expanding support for rendering with light waves. By leveraging RGBD multi-view images, our method directly optimizes complex-valued Gaussians as a 3D holographic scene representation. This eliminates the need for computationally expensive hologram re-optimization. Compared with state-of-the-art methods, our method achieves 30x-10,000x speed improvements while maintaining on-par image quality, representing a first step towards geometrically aligned, physically plausible holographic scene representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07917v1">Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Project Page: https://speede3dgs.github.io/
    </div>
    <details class="paper-abstract">
      Recent extensions of 3D Gaussian Splatting (3DGS) to dynamic scenes achieve high-quality novel view synthesis by using neural networks to predict the time-varying deformation of each Gaussian. However, performing per-Gaussian neural inference at every frame poses a significant bottleneck, limiting rendering speed and increasing memory and compute requirements. In this paper, we present Speedy Deformable 3D Gaussian Splatting (SpeeDe3DGS), a general pipeline for accelerating the rendering speed of dynamic 3DGS and 4DGS representations by reducing neural inference through two complementary techniques. First, we propose a temporal sensitivity pruning score that identifies and removes Gaussians with low contribution to the dynamic scene reconstruction. We also introduce an annealing smooth pruning mechanism that improves pruning robustness in real-world scenes with imprecise camera poses. Second, we propose GroupFlow, a motion analysis technique that clusters Gaussians by trajectory similarity and predicts a single rigid transformation per group instead of separate deformations for each Gaussian. Together, our techniques accelerate rendering by $10.37\times$, reduce model size by $7.71\times$, and shorten training time by $2.71\times$ on the NeRF-DS dataset. SpeeDe3DGS also improves rendering speed by $4.20\times$ and $58.23\times$ on the D-NeRF and HyperNeRF vrig datasets. Our methods are modular and can be integrated into any deformable 3DGS or 4DGS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07897v1">GaussianVAE: Adaptive Learning Dynamics of 3D Gaussians for High-Fidelity Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      We present a novel approach for enhancing the resolution and geometric fidelity of 3D Gaussian Splatting (3DGS) beyond native training resolution. Current 3DGS methods are fundamentally limited by their input resolution, producing reconstructions that cannot extrapolate finer details than are present in the training views. Our work breaks this limitation through a lightweight generative model that predicts and refines additional 3D Gaussians where needed most. The key innovation is our Hessian-assisted sampling strategy, which intelligently identifies regions that are likely to benefit from densification, ensuring computational efficiency. Unlike computationally intensive GANs or diffusion approaches, our method operates in real-time (0.015s per inference on a single consumer-grade GPU), making it practical for interactive applications. Comprehensive experiments demonstrate significant improvements in both geometric accuracy and rendering quality compared to state-of-the-art methods, establishing a new paradigm for resolution-free 3D scene enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07826v1">R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we will release our dataset and code, see https://research.zenseact.com/publications/R3D2/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07697v1">OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for neural scene reconstruction, offering high-quality novel view synthesis while maintaining computational efficiency. In this paper, we extend the capabilities of 3DGS beyond pure scene representation by introducing an approach for open-vocabulary 3D instance segmentation without requiring manual labeling, termed OpenSplat3D. Our method leverages feature-splatting techniques to associate semantic information with individual Gaussians, enabling fine-grained scene understanding. We incorporate Segment Anything Model instance masks with a contrastive loss formulation as guidance for the instance features to achieve accurate instance-level segmentation. Furthermore, we utilize language embeddings of a vision-language model, allowing for flexible, text-driven instance identification. This combination enables our system to identify and segment arbitrary objects in 3D scenes based on natural language descriptions. We show results on LERF-mask and LERF-OVS as well as the full ScanNet++ validation set, demonstrating the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07670v1">ProSplat: Improved Feed-Forward 3D Gaussian Splatting for Wide-Baseline Sparse Views</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Feed-forward 3D Gaussian Splatting (3DGS) has recently demonstrated promising results for novel view synthesis (NVS) from sparse input views, particularly under narrow-baseline conditions. However, its performance significantly degrades in wide-baseline scenarios due to limited texture details and geometric inconsistencies across views. To address these challenges, in this paper, we propose ProSplat, a two-stage feed-forward framework designed for high-fidelity rendering under wide-baseline conditions. The first stage involves generating 3D Gaussian primitives via a 3DGS generator. In the second stage, rendered views from these primitives are enhanced through an improvement model. Specifically, this improvement model is based on a one-step diffusion model, further optimized by our proposed Maximum Overlap Reference view Injection (MORI) and Distance-Weighted Epipolar Attention (DWEA). MORI supplements missing texture and color by strategically selecting a reference view with maximum viewpoint overlap, while DWEA enforces geometric consistency using epipolar constraints. Additionally, we introduce a divide-and-conquer training strategy that aligns data distributions between the two stages through joint optimization. We evaluate ProSplat on the RealEstate10K and DL3DV-10K datasets under wide-baseline settings. Experimental results demonstrate that ProSplat achieves an average improvement of 1 dB in PSNR compared to recent SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04120v2">Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Updated version correcting inadvertent omission in author list
    </div>
    <details class="paper-abstract">
      Creating accurate, physical simulations directly from real-world robot motion holds great value for safe, scalable, and affordable robot learning, yet remains exceptionally challenging. Real robot data suffers from occlusions, noisy camera poses, dynamic scene elements, which hinder the creation of geometrically accurate and photorealistic digital twins of unseen objects. We introduce a novel real-to-sim framework tackling all these challenges at once. Our key insight is a hybrid scene representation merging the photorealistic rendering of 3D Gaussian Splatting with explicit object meshes suitable for physics simulation within a single representation. We propose an end-to-end optimization pipeline that leverages differentiable rendering and differentiable physics within MuJoCo to jointly refine all scene components - from object geometry and appearance to robot poses and physical parameters - directly from raw and imprecise robot trajectories. This unified optimization allows us to simultaneously achieve high-fidelity object mesh reconstruction, generate photorealistic novel views, and perform annotation-free robot pose calibration. We demonstrate the effectiveness of our approach both in simulation and on challenging real-world sequences using an ALOHA 2 bi-manual manipulator, enabling more practical and robust real-to-simulation pipelines.
    </details>
</div>
