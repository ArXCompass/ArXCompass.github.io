# gaussian splatting - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07000v1">Enhancing non-Rigid 3D Model Deformations Using Mesh-based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      We propose a novel framework that enhances non-rigid 3D model deformations by bridging mesh representations with 3D Gaussian splatting. While traditional Gaussian splatting delivers fast, real-time radiance-field rendering, its post-editing capabilities and support for large-scale, non-rigid deformations remain limited. Our method addresses these challenges by embedding Gaussian kernels directly onto explicit mesh surfaces. This allows the mesh's inherent topological and geometric priors to guide intuitive editing operations -- such as moving, scaling, and rotating individual 3D components -- and enables complex deformations like bending and stretching. This work paves the way for more flexible 3D content-creation workflows in applications spanning virtual reality, character animation, and interactive design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06927v3">CULTURE3D: A Large-Scale and Diverse Dataset of Cultural Landmarks and Terrains for Gaussian-Based Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Current state-of-the-art 3D reconstruction models face limitations in building extra-large scale outdoor scenes, primarily due to the lack of sufficiently large-scale and detailed datasets. In this paper, we present a extra-large fine-grained dataset with 10 billion points composed of 41,006 drone-captured high-resolution aerial images, covering 20 diverse and culturally significant scenes from worldwide locations such as Cambridge Uni main buildings, the Pyramids, and the Forbidden City Palace. Compared to existing datasets, ours offers significantly larger scale and higher detail, uniquely suited for fine-grained 3D applications. Each scene contains an accurate spatial layout and comprehensive structural information, supporting detailed 3D reconstruction tasks. By reconstructing environments using these detailed images, our dataset supports multiple applications, including outputs in the widely adopted COLMAP format, establishing a novel benchmark for evaluating state-of-the-art large-scale Gaussian Splatting methods.The dataset's flexibility encourages innovations and supports model plug-ins, paving the way for future 3D breakthroughs. All datasets and code will be open-sourced for community use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04004v2">Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      This paper presents the first photo-realistic LiDAR-Inertial-Camera Gaussian Splatting SLAM system that simultaneously addresses visual quality, geometric accuracy, and real-time performance. The proposed method performs robust and accurate pose estimation within a continuous-time trajectory optimization framework, while incrementally reconstructing a 3D Gaussian map using camera and LiDAR data, all in real time. The resulting map enables high-quality, real-time novel view rendering of both RGB images and depth maps. To effectively address under-reconstruction in regions not covered by the LiDAR, we employ a lightweight zero-shot depth model that synergistically combines RGB appearance cues with sparse LiDAR measurements to generate dense depth maps. The depth completion enables reliable Gaussian initialization in LiDAR-blind areas, significantly improving system applicability for sparse LiDAR sensors. To enhance geometric accuracy, we use sparse but precise LiDAR depths to supervise Gaussian map optimization and accelerate it with carefully designed CUDA-accelerated strategies. Furthermore, we explore how the incrementally reconstructed Gaussian map can improve the robustness of odometry. By tightly incorporating photometric constraints from the Gaussian map into the continuous-time factor graph optimization, we demonstrate improved pose estimation under LiDAR degradation scenarios. We also showcase downstream applications via extending our elaborate system, including video frame interpolation and fast 3D mesh extraction. To support rigorous evaluation, we construct a dedicated LiDAR-Inertial-Camera dataset featuring ground-truth poses, depth maps, and extrapolated trajectories for assessing out-of-sequence novel view synthesis. Both the dataset and code will be made publicly available on project page https://xingxingzuo.github.io/gaussian_lic2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12510v2">PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Endoluminal endoscopic procedures are essential for diagnosing colorectal cancer and other severe conditions in the digestive tract, urogenital system, and airways. 3D reconstruction and novel-view synthesis from endoscopic images are promising tools for enhancing diagnosis. Moreover, integrating physiological deformations and interaction with the endoscope enables the development of simulation tools from real video data. However, constrained camera trajectories and view-dependent lighting create artifacts, leading to inaccurate or overfitted reconstructions. We present PR-ENDO, a novel 3D reconstruction framework leveraging the unique property of endoscopic imaging, where a single light source is closely aligned with the camera. Our method separates light effects from tissue properties. PR-ENDO enhances 3D Gaussian Splatting with a physically based relightable model. We boost the traditional light transport formulation with a specialized MLP capturing complex light-related effects while ensuring reduced artifacts and better generalization across novel views. PR-ENDO achieves superior reconstruction quality compared to baseline methods on both public and in-house datasets. Unlike existing approaches, PR-ENDO enables tissue modifications while preserving a physically accurate response to light, making it closer to real-world clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06684v1">Photometric Stereo using Gaussian Splatting and inverse rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 in French language. GRETSI 2025, Association GRETSI, Aug 2025, Strasbourg, France
    </div>
    <details class="paper-abstract">
      Recent state-of-the-art algorithms in photometric stereo rely on neural networks and operate either through prior learning or inverse rendering optimization. Here, we revisit the problem of calibrated photometric stereo by leveraging recent advances in 3D inverse rendering using the Gaussian Splatting formalism. This allows us to parameterize the 3D scene to be reconstructed and optimize it in a more interpretable manner. Our approach incorporates a simplified model for light representation and demonstrates the potential of the Gaussian Splatting rendering engine for the photometric stereo problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06671v1">FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 To appear at ACM MM 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has become a prominent technique for representing and rendering complex 3D scenes, due to its high fidelity and speed advantages. However, the growing demand for large-scale models calls for effective compression to reduce memory and computation costs, especially on mobile and edge devices with limited resources. Existing compression methods effectively reduce 3D Gaussian parameters but often require extensive retraining or fine-tuning, lacking flexibility under varying compression constraints. In this paper, we introduce FlexGaussian, a flexible and cost-effective method that combines mixed-precision quantization with attribute-discriminative pruning for training-free 3D Gaussian compression. FlexGaussian eliminates the need for retraining and adapts easily to diverse compression targets. Evaluation results show that FlexGaussian achieves up to 96.4% compression while maintaining high rendering quality (<1 dB drop in PSNR), and is deployable on mobile devices. FlexGaussian delivers high compression ratios within seconds, being 1.7-2.1x faster than state-of-the-art training-free methods and 10-100x faster than training-involved approaches. The code is being prepared and will be released soon at: https://github.com/Supercomputing-System-AI-Lab/FlexGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06647v1">ClipGS: Clippable Gaussian Splatting for Interactive Cinematic Visualization of Volumetric Medical Data</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Early accepted by MICCAI 2025. Project is available at: https://med-air.github.io/ClipGS
    </div>
    <details class="paper-abstract">
      The visualization of volumetric medical data is crucial for enhancing diagnostic accuracy and improving surgical planning and education. Cinematic rendering techniques significantly enrich this process by providing high-quality visualizations that convey intricate anatomical details, thereby facilitating better understanding and decision-making in medical contexts. However, the high computing cost and low rendering speed limit the requirement of interactive visualization in practical applications. In this paper, we introduce ClipGS, an innovative Gaussian splatting framework with the clipping plane supported, for interactive cinematic visualization of volumetric medical data. To address the challenges posed by dynamic interactions, we propose a learnable truncation scheme that automatically adjusts the visibility of Gaussian primitives in response to the clipping plane. Besides, we also design an adaptive adjustment model to dynamically adjust the deformation of Gaussians and refine the rendering performance. We validate our method on five volumetric medical data (including CT and anatomical slice data), and reach an average 36.635 PSNR rendering quality with 156 FPS and 16.1 MB model size, outperforming state-of-the-art methods in rendering quality and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18133v2">Gaussian Fluids: A Grid-Free Fluid Solver based on Gaussian Spatial Representation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      We present a grid-free fluid solver featuring a novel Gaussian representation. Drawing inspiration from the expressive capabilities of 3D Gaussian Splatting in multi-view image reconstruction, we model the continuous flow velocity as a weighted sum of multiple Gaussian functions. This representation is continuously differentiable, which enables us to derive spatial differentials directly and solve the time-dependent PDE via a custom first-order optimization tailored to fluid dynamics. Compared to traditional discretizations, which typically adopt Eulerian, Lagrangian, or hybrid perspectives, our approach is inherently memory-efficient and spatially adaptive, enabling it to preserve fine-scale structures and vortices with high fidelity. While these advantages are also sought by implicit neural representations, GSR offers enhanced robustness, accuracy, and generality across diverse fluid phenomena, with improved computational efficiency during temporal evolution. Though our first-order solver does not yet match the speed of fluid solvers using explicit representations, its continuous nature substantially reduces spatial discretization error and opens a new avenue for high-fidelity simulation. We evaluate the proposed solver across a broad range of 2D and 3D fluid phenomena, demonstrating its ability to preserve intricate vortex dynamics, accurately capture boundary-induced effects such as K\'arm\'an vortex streets, and remain robust across long time horizons - all without additional parameter tuning. Our results suggest that GSR offers a compelling direction for future research in fluid simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18394v4">Reconstructing Satellites in 3D from Amateur Telescope Images</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Monitoring space objects is crucial for space situational awareness, yet reconstructing 3D satellite models from ground-based telescope images is challenging due to atmospheric turbulence, long observation distances, limited viewpoints, and low signal-to-noise ratios. In this paper, we propose a novel computational imaging framework that overcomes these obstacles by integrating a hybrid image pre-processing pipeline with a joint pose estimation and 3D reconstruction module based on controlled Gaussian Splatting (GS) and Branch-and-Bound (BnB) search. We validate our approach on both synthetic satellite datasets and on-sky observations of China's Tiangong Space Station and the International Space Station, achieving robust 3D reconstructions of low-Earth orbit satellites from ground-based data. Quantitative evaluations using SSIM, PSNR, LPIPS, and Chamfer Distance demonstrate that our method outperforms state-of-the-art NeRF-based approaches, and ablation studies confirm the critical role of each component. Our framework enables high-fidelity 3D satellite monitoring from Earth, offering a cost-effective alternative for space situational awareness. Project page: https://ai4scientificimaging.org/ReconstructingSatellites
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15582v2">EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Acccpeted by ICCV2025
    </div>
    <details class="paper-abstract">
      Photorealistic reconstruction of street scenes is essential for developing real-world simulators in autonomous driving. While recent methods based on 3D/4D Gaussian Splatting (GS) have demonstrated promising results, they still encounter challenges in complex street scenes due to the unpredictable motion of dynamic objects. Current methods typically decompose street scenes into static and dynamic objects, learning the Gaussians in either a supervised manner (e.g., w/ 3D bounding-box) or a self-supervised manner (e.g., w/o 3D bounding-box). However, these approaches do not effectively model the motions of dynamic objects (e.g., the motion speed of pedestrians is clearly different from that of vehicles), resulting in suboptimal scene decomposition. To address this, we propose Explicit Motion Decomposition (EMD), which models the motions of dynamic objects by introducing learnable motion embeddings to the Gaussians, enhancing the decomposition in street scenes. The proposed plug-and-play EMD module compensates for the lack of motion modeling in self-supervised street Gaussian splatting methods. We also introduce tailored training strategies to extend EMD to supervised approaches. Comprehensive experiments demonstrate the effectiveness of our method, achieving state-of-the-art novel view synthesis performance in self-supervised settings. The code is available at: https://qingpowuwu.github.io/emd.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12981v2">GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted by ICCV2025
    </div>
    <details class="paper-abstract">
      Gaze estimation encounters generalization challenges when dealing with out-of-distribution data. To address this problem, recent methods use neural radiance fields (NeRF) to generate augmented data. However, existing methods based on NeRF are computationally expensive and lack facial details. 3D Gaussian Splatting (3DGS) has become the prevailing representation of neural fields. While 3DGS has been extensively examined in head avatars, it faces challenges with accurate gaze control and generalization across different subjects. In this work, we propose GazeGaussian, the first high-fidelity gaze redirection method that uses a two-stream 3DGS model to represent the face and eye regions separately. Leveraging the unstructured nature of 3DGS, we develop a novel representation of the eye for rigid eye rotation based on the target gaze direction. To enable synthesis generalization across various subjects, we integrate an expression-guided module to inject subject-specific information into the neural renderer. Comprehensive experiments show that GazeGaussian outperforms existing methods in rendering speed, gaze redirection accuracy, and facial synthesis across multiple datasets. The code is available at: https://ucwxb.github.io/GazeGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14931v2">Hier-SLAM++: Neuro-Symbolic Semantic SLAM with a Hierarchically Categorical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 18 pages. Under review
    </div>
    <details class="paper-abstract">
      We propose Hier-SLAM++, a comprehensive Neuro-Symbolic semantic 3D Gaussian Splatting SLAM method with both RGB-D and monocular input featuring an advanced hierarchical categorical representation, which enables accurate pose estimation as well as global 3D semantic mapping. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making scene understanding particularly challenging and costly. To address this problem, we introduce a novel hierarchical representation that encodes both semantic and geometric information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs) as well as the 3D generative model. By utilizing the proposed hierarchical tree structure, semantic information is symbolically represented and learned in an end-to-end manner. We further introduce an advanced semantic loss designed to optimize hierarchical semantic information through both Intra-level and Inter-level optimizations. Additionally, we propose an improved SLAM system to support both RGB-D and monocular inputs using a feed-forward model. To the best of our knowledge, this is the first semantic monocular Gaussian Splatting SLAM system, significantly reducing sensor requirements for 3D semantic understanding and broadening the applicability of semantic Gaussian SLAM system. We conduct experiments on both synthetic and real-world datasets, demonstrating superior or on-par performance with state-of-the-art methods, while significantly reducing storage and training time requirements. Our project page is available at: https://hierslampp.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02803v2">HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Project page: https://gserifi.github.io/HyperGaussians, Code: https://github.com/gserifi/HyperGaussians
    </div>
    <details class="paper-abstract">
      We introduce HyperGaussians, a novel extension of 3D Gaussian Splatting for high-quality animatable face avatars. Creating such detailed face avatars from videos is a challenging problem and has numerous applications in augmented and virtual reality. While tremendous successes have been achieved for static faces, animatable avatars from monocular videos still fall in the uncanny valley. The de facto standard, 3D Gaussian Splatting (3DGS), represents a face through a collection of 3D Gaussian primitives. 3DGS excels at rendering static faces, but the state-of-the-art still struggles with nonlinear deformations, complex lighting effects, and fine details. While most related works focus on predicting better Gaussian parameters from expression codes, we rethink the 3D Gaussian representation itself and how to make it more expressive. Our insights lead to a novel extension of 3D Gaussians to high-dimensional multivariate Gaussians, dubbed 'HyperGaussians'. The higher dimensionality increases expressivity through conditioning on a learnable local embedding. However, splatting HyperGaussians is computationally expensive because it requires inverting a high-dimensional covariance matrix. We solve this by reparameterizing the covariance matrix, dubbed the 'inverse covariance trick'. This trick boosts the efficiency so that HyperGaussians can be seamlessly integrated into existing models. To demonstrate this, we plug in HyperGaussians into the state-of-the-art in fast monocular face avatars: FlashAvatar. Our evaluation on 19 subjects from 4 face datasets shows that HyperGaussians outperform 3DGS numerically and visually, particularly for high-frequency details like eyeglass frames, teeth, complex facial movements, and specular reflections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06109v1">LighthouseGS: Indoor Structure-aware 3D Gaussian Splatting for Panorama-Style Mobile Captures</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time novel view synthesis (NVS) with impressive quality in indoor scenes. However, achieving high-fidelity rendering requires meticulously captured images covering the entire scene, limiting accessibility for general users. We aim to develop a practical 3DGS-based NVS framework using simple panorama-style motion with a handheld camera (e.g., mobile device). While convenient, this rotation-dominant motion and narrow baseline make accurate camera pose and 3D point estimation challenging, especially in textureless indoor scenes. To address these challenges, we propose LighthouseGS, a novel framework inspired by the lighthouse-like sweeping motion of panoramic views. LighthouseGS leverages rough geometric priors, such as mobile device camera poses and monocular depth estimation, and utilizes the planar structures often found in indoor environments. We present a new initialization method called plane scaffold assembly to generate consistent 3D points on these structures, followed by a stable pruning strategy to enhance geometry and optimization stability. Additionally, we introduce geometric and photometric corrections to resolve inconsistencies from motion drift and auto-exposure in mobile devices. Tested on collected real and synthetic indoor scenes, LighthouseGS delivers photorealistic rendering, surpassing state-of-the-art methods and demonstrating the potential for panoramic view synthesis and object placement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06103v1">Reflections Unlock: Geometry-Aware Reflection Disentanglement in 3D Gaussian Splatting for Photorealistic Scenes Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Accurately rendering scenes with reflective surfaces remains a significant challenge in novel view synthesis, as existing methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) often misinterpret reflections as physical geometry, resulting in degraded reconstructions. Previous methods rely on incomplete and non-generalizable geometric constraints, leading to misalignment between the positions of Gaussian splats and the actual scene geometry. When dealing with real-world scenes containing complex geometry, the accumulation of Gaussians further exacerbates surface artifacts and results in blurred reconstructions. To address these limitations, in this work, we propose Ref-Unlock, a novel geometry-aware reflection modeling framework based on 3D Gaussian Splatting, which explicitly disentangles transmitted and reflected components to better capture complex reflections and enhance geometric consistency in real-world scenes. Our approach employs a dual-branch representation with high-order spherical harmonics to capture high-frequency reflective details, alongside a reflection removal module providing pseudo reflection-free supervision to guide clean decomposition. Additionally, we incorporate pseudo-depth maps and a geometry-aware bilateral smoothness constraint to enhance 3D geometric consistency and stability in decomposition. Extensive experiments demonstrate that Ref-Unlock significantly outperforms classical GS-based reflection methods and achieves competitive results with NeRF-based models, while enabling flexible vision foundation models (VFMs) driven reflection editing. Our method thus offers an efficient and generalizable solution for realistic rendering of reflective scenes. Our code is available at https://ref-unlock.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06060v1">VisualSpeaker: Visually-Guided 3D Avatar Lip Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Realistic, high-fidelity 3D facial animations are crucial for expressive avatar systems in human-computer interaction and accessibility. Although prior methods show promising quality, their reliance on the mesh domain limits their ability to fully leverage the rapid visual innovations seen in 2D computer vision and graphics. We propose VisualSpeaker, a novel method that bridges this gap using photorealistic differentiable rendering, supervised by visual speech recognition, for improved 3D facial animation. Our contribution is a perceptual lip-reading loss, derived by passing photorealistic 3D Gaussian Splatting avatar renders through a pre-trained Visual Automatic Speech Recognition model during training. Evaluation on the MEAD dataset demonstrates that VisualSpeaker improves both the standard Lip Vertex Error metric by 56.1% and the perceptual quality of the generated animations, while retaining the controllability of mesh-driven animation. This perceptual focus naturally supports accurate mouthings, essential cues that disambiguate similar manual signs in sign language avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22099v3">BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted at ICCV 2025, Project Page: https://github.com/fudan-zvg/BezierGS
    </div>
    <details class="paper-abstract">
      The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose B\'ezier curve Gaussian splatting (B\'ezierGS), which represents the motion trajectories of dynamic objects using learnable B\'ezier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that B\'ezierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05859v1">D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 12 pages, 9 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Free-viewpoint video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representations remains a major challenge. Recent advances in 3D Gaussian Splatting (3DGS) and its dynamic extensions have enabled high-fidelity scene modeling. However, existing methods often couple scene reconstruction with optimization-dependent coding, which limits generalizability. This paper presents Feedforward Compression of Dynamic Gaussian Splatting (D-FCGS), a novel feedforward framework for compressing temporally correlated Gaussian point cloud sequences. Our approach introduces a Group-of-Frames (GoF) structure with I-P frame coding, where inter-frame motions are extracted via sparse control points. The resulting motion tensors are compressed in a feedforward manner using a dual prior-aware entropy model that combines hyperprior and spatial-temporal priors for accurate rate estimation. For reconstruction, we perform control-point-guided motion compensation and employ a refinement network to enhance view-consistent fidelity. Trained on multi-view video-derived Gaussian frames, D-FCGS generalizes across scenes without per-scene optimization. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 40 times compression in under 2 seconds while preserving visual quality across viewpoints. This work advances feedforward compression for dynamic 3DGS, paving the way for scalable FVV transmission and storage in immersive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05763v1">DreamArt: Generating Interactable Articulated Objects from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Generating articulated objects, such as laptops and microwaves, is a crucial yet challenging task with extensive applications in Embodied AI and AR/VR. Current image-to-3D methods primarily focus on surface geometry and texture, neglecting part decomposition and articulation modeling. Meanwhile, neural reconstruction approaches (e.g., NeRF or Gaussian Splatting) rely on dense multi-view or interaction data, limiting their scalability. In this paper, we introduce DreamArt, a novel framework for generating high-fidelity, interactable articulated assets from single-view images. DreamArt employs a three-stage pipeline: firstly, it reconstructs part-segmented and complete 3D object meshes through a combination of image-to-3D generation, mask-prompted 3D segmentation, and part amodal completion. Second, we fine-tune a video diffusion model to capture part-level articulation priors, leveraging movable part masks as prompt and amodal images to mitigate ambiguities caused by occlusion. Finally, DreamArt optimizes the articulation motion, represented by a dual quaternion, and conducts global texture refinement and repainting to ensure coherent, high-quality textures across all parts. Experimental results demonstrate that DreamArt effectively generates high-quality articulated objects, possessing accurate part shape, high appearance fidelity, and plausible articulation, thereby providing a scalable solution for articulated asset generation. Our project page is available at https://dream-art-0.github.io/DreamArt/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05661v1">3DGS_LSR:Large_Scale Relocation for Autonomous Driving Based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 13 pages,7 figures,4 tables
    </div>
    <details class="paper-abstract">
      In autonomous robotic systems, precise localization is a prerequisite for safe navigation. However, in complex urban environments, GNSS positioning often suffers from signal occlusion and multipath effects, leading to unreliable absolute positioning. Traditional mapping approaches are constrained by storage requirements and computational inefficiency, limiting their applicability to resource-constrained robotic platforms. To address these challenges, we propose 3DGS-LSR: a large-scale relocalization framework leveraging 3D Gaussian Splatting (3DGS), enabling centimeter-level positioning using only a single monocular RGB image on the client side. We combine multi-sensor data to construct high-accuracy 3DGS maps in large outdoor scenes, while the robot-side localization requires just a standard camera input. Using SuperPoint and SuperGlue for feature extraction and matching, our core innovation is an iterative optimization strategy that refines localization results through step-by-step rendering, making it suitable for real-time autonomous navigation. Experimental validation on the KITTI dataset demonstrates our 3DGS-LSR achieves average positioning accuracies of 0.026m, 0.029m, and 0.081m in town roads, boulevard roads, and traffic-dense highways respectively, significantly outperforming other representative methods while requiring only monocular RGB input. This approach provides autonomous robots with reliable localization capabilities even in challenging urban environments where GNSS fails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06269v1">A Probabilistic Approach to Uncertainty Quantification Leveraging 3D Geometry</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 ICCV 2025 Workshops (8 Pages, 6 Figures, 2 Tables)
    </div>
    <details class="paper-abstract">
      Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and awareness of fidelity surface geometric uncertainty are essential. Unlike radiance-based models such as NeRF or 3D Gaussian splatting, which lack explicit surface formulations, SDFs define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability via Hessian-based metrics, enabling computationally efficient, surface-aware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05256v1">SegmentDreamer: Towards High-fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted by ICCV 2025, project page: https://zjhjojo.github.io/
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation improve the visual quality of Score Distillation Sampling (SDS) and its variants by directly connecting Consistency Distillation (CD) to score distillation. However, due to the imbalance between self-consistency and cross-consistency, these CD-based methods inherently suffer from improper conditional guidance, leading to sub-optimal generation results. To address this issue, we present SegmentDreamer, a novel framework designed to fully unleash the potential of consistency models for high-fidelity text-to-3D generation. Specifically, we reformulate SDS through the proposed Segmented Consistency Trajectory Distillation (SCTD), effectively mitigating the imbalance issues by explicitly defining the relationship between self- and cross-consistency. Moreover, SCTD partitions the Probability Flow Ordinary Differential Equation (PF-ODE) trajectory into multiple sub-trajectories and ensures consistency within each segment, which can theoretically provide a significantly tighter upper bound on distillation error. Additionally, we propose a distillation pipeline for a more swift and stable generation. Extensive experiments demonstrate that our SegmentDreamer outperforms state-of-the-art methods in visual quality, enabling high-fidelity 3D asset creation through 3D Gaussian Splatting (3DGS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05040v3">GaussRender: Learning 3D Occupancy with Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Understanding the 3D geometry and semantics of driving scenes is critical for safe autonomous driving. Recent advances in 3D occupancy prediction have improved scene representation but often suffer from visual inconsistencies, leading to floating artifacts and poor surface localization. Existing voxel-wise losses (e.g., cross-entropy) fail to enforce visible geometric coherence. In this paper, we propose GaussRender, a module that improves 3D occupancy learning by enforcing projective consistency. Our key idea is to project both predicted and ground-truth 3D occupancy into 2D camera views, where we apply supervision. Our method penalizes 3D configurations that produce inconsistent 2D projections, thereby enforcing a more coherent 3D structure. To achieve this efficiently, we leverage differentiable rendering with Gaussian splatting. GaussRender seamlessly integrates with existing architectures while maintaining efficiency and requiring no inference-time modifications. Extensive evaluations on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate that GaussRender significantly improves geometric fidelity across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), achieving state-of-the-art results, particularly on surface-sensitive metrics such as RayIoU. The code is open-sourced at https://github.com/valeoai/GaussRender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04961v1">InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting based 3D editing has demonstrated impressive performance in recent years. However, the multi-view editing often exhibits significant local inconsistency, especially in areas of non-rigid deformation, which lead to local artifacts, texture blurring, or semantic variations in edited 3D scenes. We also found that the existing editing methods, which rely entirely on text prompts make the editing process a "one-shot deal", making it difficult for users to control the editing degree flexibly. In response to these challenges, we present InterGSEdit, a novel framework for high-quality 3DGS editing via interactively selecting key views with users' preferences. We propose a CLIP-based Semantic Consistency Selection (CSCS) strategy to adaptively screen a group of semantically consistent reference views for each user-selected key view. Then, the cross-attention maps derived from the reference views are used in a weighted Gaussian Splatting unprojection to construct the 3D Geometry-Consistent Attention Prior ($GAP^{3D}$). We project $GAP^{3D}$ to obtain 3D-constrained attention, which are fused with 2D cross-attention via Attention Fusion Network (AFN). AFN employs an adaptive attention strategy that prioritizes 3D-constrained attention for geometric consistency during early inference, and gradually prioritizes 2D cross-attention maps in diffusion for fine-grained features during the later inference. Extensive experiments demonstrate that InterGSEdit achieves state-of-the-art performance, delivering consistent, high-fidelity 3DGS editing with improved user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05426v1">Mastering Regional 3DGS: Locating, Initializing, and Editing with Diverse 2D Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Many 3D scene editing tasks focus on modifying local regions rather than the entire scene, except for some global applications like style transfer, and in the context of 3D Gaussian Splatting (3DGS), where scenes are represented by a series of Gaussians, this structure allows for precise regional edits, offering enhanced control over specific areas of the scene; however, the challenge lies in the fact that 3D semantic parsing often underperforms compared to its 2D counterpart, making targeted manipulations within 3D spaces more difficult and limiting the fidelity of edits, which we address by leveraging 2D diffusion editing to accurately identify modification regions in each view, followed by inverse rendering for 3D localization, then refining the frontal view and initializing a coarse 3DGS with consistent views and approximate shapes derived from depth maps predicted by a 2D foundation model, thereby supporting an iterative, view-consistent editing process that gradually enhances structural details and textures to ensure coherence across perspectives. Experiments demonstrate that our method achieves state-of-the-art performance while delivering up to a $4\times$ speedup, providing a more efficient and effective approach to 3D scene local editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19420v3">RF-3DGS: Wireless Channel Modeling with Radio Radiance Field and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 in submission to IEEE journals
    </div>
    <details class="paper-abstract">
      Precisely modeling radio propagation in complex environments has been a significant challenge, especially with the advent of 5G and beyond networks, where managing massive antenna arrays demands more detailed information. Traditional methods, such as empirical models and ray tracing, often fall short, either due to insufficient details or because of challenges for real-time applications. Inspired by the newly proposed 3D Gaussian Splatting method in the computer vision domain, which outperforms other methods in reconstructing optical radiance fields, we propose RF-3DGS, a novel approach that enables precise site-specific reconstruction of radio radiance fields from sparse samples. RF-3DGS can render radio spatial spectra at arbitrary positions within 2 ms following a brief 3-minute training period, effectively identifying dominant propagation paths. Furthermore, RF-3DGS can provide fine-grained Spatial Channel State Information (Spatial-CSI) of these paths, including the channel gain, the delay, the angle of arrival (AoA), and the angle of departure (AoD). Our experiments, calibrated through real-world measurements, demonstrate that RF-3DGS not only significantly improves reconstruction quality, training efficiency, and rendering speed compared to state-of-the-art methods, but also holds great potential for supporting wireless communication and advanced applications such as Integrated Sensing and Communication (ISAC). Code and dataset will be available at https://github.com/SunLab-UGA/RF-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00554v2">LOD-GS: Level-of-Detail-Sensitive 3D Gaussian Splatting for Detail Conserved Anti-Aliasing</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Despite the advancements in quality and efficiency achieved by 3D Gaussian Splatting (3DGS) in 3D scene rendering, aliasing artifacts remain a persistent challenge. Existing approaches primarily rely on low-pass filtering to mitigate aliasing. However, these methods are not sensitive to the sampling rate, often resulting in under-filtering and over-smoothing renderings. To address this limitation, we propose LOD-GS, a Level-of-Detail-sensitive filtering framework for Gaussian Splatting, which dynamically predicts the optimal filtering strength for each 3D Gaussian primitive. Specifically, we introduce a set of basis functions to each Gaussian, which take the sampling rate as input to model appearance variations, enabling sampling-rate-sensitive filtering. These basis function parameters are jointly optimized with the 3D Gaussian in an end-to-end manner. The sampling rate is influenced by both focal length and camera distance. However, existing methods and datasets rely solely on down-sampling to simulate focal length changes for anti-aliasing evaluation, overlooking the impact of camera distance. To enable a more comprehensive assessment, we introduce a new synthetic dataset featuring objects rendered at varying camera distances. Extensive experiments on both public datasets and our newly collected dataset demonstrate that our method achieves SOTA rendering quality while effectively eliminating aliasing. The code and dataset have been open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04147v1">A3FR: Agile 3D Gaussian Splatting with Incremental Gaze Tracked Foveated Rendering in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 ACM International Conference on Supercomputing 2025
    </div>
    <details class="paper-abstract">
      Virtual reality (VR) significantly transforms immersive digital interfaces, greatly enhancing education, professional practices, and entertainment by increasing user engagement and opening up new possibilities in various industries. Among its numerous applications, image rendering is crucial. Nevertheless, rendering methodologies like 3D Gaussian Splatting impose high computational demands, driven predominantly by user expectations for superior visual quality. This results in notable processing delays for real-time image rendering, which greatly affects the user experience. Additionally, VR devices such as head-mounted displays (HMDs) are intricately linked to human visual behavior, leveraging knowledge from perception and cognition to improve user experience. These insights have spurred the development of foveated rendering, a technique that dynamically adjusts rendering resolution based on the user's gaze direction. The resultant solution, known as gaze-tracked foveated rendering, significantly reduces the computational burden of the rendering process. Although gaze-tracked foveated rendering can reduce rendering costs, the computational overhead of the gaze tracking process itself can sometimes outweigh the rendering savings, leading to increased processing latency. To address this issue, we propose an efficient rendering framework called~\textit{A3FR}, designed to minimize the latency of gaze-tracked foveated rendering via the parallelization of gaze tracking and foveated rendering processes. For the rendering algorithm, we utilize 3D Gaussian Splatting, a state-of-the-art neural rendering technique. Evaluation results demonstrate that A3FR can reduce end-to-end rendering latency by up to $2\times$ while maintaining visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01110v2">A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks -- a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously. We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU -- without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to support real-time streaming and rendering. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes -- from broad aerial views to fine-grained ground-level details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11868v2">MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Accurately analyzing the motion parts and their motion attributes in dynamic environments is crucial for advancing key areas such as embodied intelligence. Addressing the limitations of existing methods that rely on dense multi-view images or detailed part-level annotations, we propose an innovative framework that can analyze 3D mobility from monocular videos in a zero-shot manner. This framework can precisely parse motion parts and motion attributes only using a monocular video, completely eliminating the need for annotated training data. Specifically, our method first constructs the scene geometry and roughly analyzes the motion parts and their initial motion attributes combining depth estimation, optical flow analysis and point cloud registration method, then employs 2D Gaussian splatting for scene representation. Building on this, we introduce an end-to-end dynamic scene optimization algorithm specifically designed for articulated objects, refining the initial analysis results to ensure the system can handle 'rotation', 'translation', and even complex movements ('rotation+translation'), demonstrating high flexibility and versatility. To validate the robustness and wide applicability of our method, we created a comprehensive dataset comprising both simulated and real-world scenarios. Experimental results show that our framework can effectively analyze articulated object motions in an annotation-free manner, showcasing its significant potential in future embodied intelligence applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04004v1">Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      This paper proposes an innovative LiDAR-Inertial-Camera SLAM system with 3D Gaussian Splatting, which is the first to jointly consider visual quality, geometric accuracy, and real-time performance. It robustly and accurately estimates poses while building a photo-realistic 3D Gaussian map in real time that enables high-quality novel view RGB and depth rendering. To effectively address under-reconstruction in regions not covered by the LiDAR, we employ a lightweight zero-shot depth model that synergistically combines RGB appearance cues with sparse LiDAR measurements to generate dense depth maps. The depth completion enables reliable Gaussian initialization in LiDAR-blind areas, significantly improving system applicability for sparse LiDAR sensors. To enhance geometric accuracy, we use sparse but precise LiDAR depths to supervise Gaussian map optimization and accelerate it with carefully designed CUDA-accelerated strategies. Furthermore, we explore how the incrementally reconstructed Gaussian map can improve the robustness of odometry. By tightly incorporating photometric constraints from the Gaussian map into the continuous-time factor graph optimization, we demonstrate improved pose estimation under LiDAR degradation scenarios. We also showcase downstream applications via extending our elaborate system, including video frame interpolation and fast 3D mesh extraction. To support rigorous evaluation, we construct a dedicated LiDAR-Inertial-Camera dataset featuring ground-truth poses, depth maps, and extrapolated trajectories for assessing out-of-sequence novel view synthesis. Extensive experiments on both public and self-collected datasets demonstrate the superiority and versatility of our system across LiDAR sensors with varying sampling densities. Both the dataset and code will be made publicly available on project page https://xingxingzuo.github.io/gaussian_lic2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21420v2">EndoFlow-SLAM: Real-Time Endoscopic SLAM with Flow-Constrained Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 This paper has been accepted at MICCAI2025
    </div>
    <details class="paper-abstract">
      Efficient three-dimensional reconstruction and real-time visualization are critical in surgical scenarios such as endoscopy. In recent years, 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in efficient 3D reconstruction and rendering. Most 3DGS-based Simultaneous Localization and Mapping (SLAM) methods only rely on the appearance constraints for optimizing both 3DGS and camera poses. However, in endoscopic scenarios, the challenges include photometric inconsistencies caused by non-Lambertian surfaces and dynamic motion from breathing affects the performance of SLAM systems. To address these issues, we additionally introduce optical flow loss as a geometric constraint, which effectively constrains both the 3D structure of the scene and the camera motion. Furthermore, we propose a depth regularisation strategy to mitigate the problem of photometric inconsistencies and ensure the validity of 3DGS depth rendering in endoscopic scenes. In addition, to improve scene representation in the SLAM system, we improve the 3DGS refinement strategy by focusing on viewpoints corresponding to Keyframes with suboptimal rendering quality frames, achieving better rendering results. Extensive experiments on the C3VD static dataset and the StereoMIS dynamic dataset demonstrate that our method outperforms existing state-of-the-art methods in novel view synthesis and pose estimation, exhibiting high performance in both static and dynamic surgical scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03886v1">ArmGS: Composite Gaussian Appearance Refinement for Modeling Dynamic Urban Environments</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      This work focuses on modeling dynamic urban environments for autonomous driving simulation. Contemporary data-driven methods using neural radiance fields have achieved photorealistic driving scene modeling, but they suffer from low rendering efficacy. Recently, some approaches have explored 3D Gaussian splatting for modeling dynamic urban scenes, enabling high-fidelity reconstruction and real-time rendering. However, these approaches often neglect to model fine-grained variations between frames and camera viewpoints, leading to suboptimal results. In this work, we propose a new approach named ArmGS that exploits composite driving Gaussian splatting with multi-granularity appearance refinement for autonomous driving scene modeling. The core idea of our approach is devising a multi-level appearance modeling scheme to optimize a set of transformation parameters for composite Gaussian refinement from multiple granularities, ranging from local Gaussian level to global image level and dynamic actor level. This not only models global scene appearance variations between frames and camera viewpoints, but also models local fine-grained changes of background and objects. Extensive experiments on multiple challenging autonomous driving datasets, namely, Waymo, KITTI, NOTR and VKITTI2, demonstrate the superiority of our approach over the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03737v1">Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted by ICCV2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become a popular solution in SLAM due to its high-fidelity and real-time novel view synthesis performance. However, some previous 3DGS SLAM methods employ a differentiable rendering pipeline for tracking, \textbf{lack geometric priors} in outdoor scenes. Other approaches introduce separate tracking modules, but they accumulate errors with significant camera movement, leading to \textbf{scale drift}. To address these challenges, we propose a robust RGB-only outdoor 3DGS SLAM method: S3PO-GS. Technically, we establish a self-consistent tracking module anchored in the 3DGS pointmap, which avoids cumulative scale drift and achieves more precise and robust tracking with fewer iterations. Additionally, we design a patch-based pointmap dynamic mapping module, which introduces geometric priors while avoiding scale ambiguity. This significantly enhances tracking accuracy and the quality of scene reconstruction, making it particularly suitable for complex outdoor environments. Our experiments on the Waymo, KITTI, and DL3DV datasets demonstrate that S3PO-GS achieves state-of-the-art results in novel view synthesis and outperforms other 3DGS SLAM methods in tracking accuracy. Project page: https://3dagentworld.github.io/S3PO-GS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08135v2">ArticulatedGS: Self-supervised Digital Twin Modeling of Articulated Objects using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      We tackle the challenge of concurrent reconstruction at the part level with the RGB appearance and estimation of motion parameters for building digital twins of articulated objects using the 3D Gaussian Splatting (3D-GS) method. With two distinct sets of multi-view imagery, each depicting an object in separate static articulation configurations, we reconstruct the articulated object in 3D Gaussian representations with both appearance and geometry information at the same time. Our approach decoupled multiple highly interdependent parameters through a multi-step optimization process, thereby achieving a stable optimization procedure and high-quality outcomes. We introduce ArticulatedGS, a self-supervised, comprehensive framework that autonomously learns to model shapes and appearances at the part level and synchronizes the optimization of motion parameters, all without reliance on 3D supervision, motion cues, or semantic labels. Our experimental results demonstrate that, among comparable methodologies, our approach has achieved optimal outcomes in terms of part segmentation accuracy, motion estimation accuracy, and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02803v1">HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Project page: https://gserifi.github.io/HyperGaussians
    </div>
    <details class="paper-abstract">
      We introduce HyperGaussians, a novel extension of 3D Gaussian Splatting for high-quality animatable face avatars. Creating such detailed face avatars from videos is a challenging problem and has numerous applications in augmented and virtual reality. While tremendous successes have been achieved for static faces, animatable avatars from monocular videos still fall in the uncanny valley. The de facto standard, 3D Gaussian Splatting (3DGS), represents a face through a collection of 3D Gaussian primitives. 3DGS excels at rendering static faces, but the state-of-the-art still struggles with nonlinear deformations, complex lighting effects, and fine details. While most related works focus on predicting better Gaussian parameters from expression codes, we rethink the 3D Gaussian representation itself and how to make it more expressive. Our insights lead to a novel extension of 3D Gaussians to high-dimensional multivariate Gaussians, dubbed 'HyperGaussians'. The higher dimensionality increases expressivity through conditioning on a learnable local embedding. However, splatting HyperGaussians is computationally expensive because it requires inverting a high-dimensional covariance matrix. We solve this by reparameterizing the covariance matrix, dubbed the 'inverse covariance trick'. This trick boosts the efficiency so that HyperGaussians can be seamlessly integrated into existing models. To demonstrate this, we plug in HyperGaussians into the state-of-the-art in fast monocular face avatars: FlashAvatar. Our evaluation on 19 subjects from 4 face datasets shows that HyperGaussians outperform 3DGS numerically and visually, particularly for high-frequency details like eyeglass frames, teeth, complex facial movements, and specular reflections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05511v5">Free Your Hands: Lightweight Turntable-Based Object Capture Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) from multiple captured photos of an object is a widely studied problem. Achieving high quality typically requires dense sampling of input views, which can lead to frustrating manual labor. Manually positioning cameras to maintain an optimal desired distribution can be difficult for humans, and if a good distribution is found, it is not easy to replicate. Additionally, the captured data can suffer from motion blur and defocus due to human error. In this paper, we use a lightweight object capture pipeline to reduce the manual workload and standardize the acquisition setup, with a consumer turntable to carry the object and a tripod to hold the camera. Of course, turntables and gantry systems have been frequently used to automatically capture dense samples under various views and lighting conditions; the key difference is that we use a turntable under natural environment lighting. This way, we can easily capture hundreds of valid images in several minutes without hands-on effort. However, in the object reference frame, the light conditions vary (rotate); this does not match the assumptions of standard NVS methods like 3D Gaussian splatting (3DGS). We design a neural radiance representation conditioned on light rotations, which addresses this issue and allows rendering with novel light rotations as an additional benefit. We further study the behavior of rotations and find optimal capturing strategies. We demonstrate our pipeline using 3DGS as the underlying framework, achieving higher quality and showcasing the method's potential for novel lighting and harmonization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02600v1">ArtGS:3D Gaussian Splatting for Interactive Visual-Physical Modeling and Manipulation of Articulated Objects</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      Articulated object manipulation remains a critical challenge in robotics due to the complex kinematic constraints and the limited physical reasoning of existing methods. In this work, we introduce ArtGS, a novel framework that extends 3D Gaussian Splatting (3DGS) by integrating visual-physical modeling for articulated object understanding and interaction. ArtGS begins with multi-view RGB-D reconstruction, followed by reasoning with a vision-language model (VLM) to extract semantic and structural information, particularly the articulated bones. Through dynamic, differentiable 3DGS-based rendering, ArtGS optimizes the parameters of the articulated bones, ensuring physically consistent motion constraints and enhancing the manipulation policy. By leveraging dynamic Gaussian splatting, cross-embodiment adaptability, and closed-loop optimization, ArtGS establishes a new framework for efficient, scalable, and generalizable articulated object modeling and manipulation. Experiments conducted in both simulation and real-world environments demonstrate that ArtGS significantly outperforms previous methods in joint estimation accuracy and manipulation success rates across a variety of articulated objects. Additional images and videos are available on the project website: https://sites.google.com/view/artgs/home
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22800v2">RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      A single-pass driving clip frequently results in incomplete scanning of the road structure, making reconstructed scene expanding a critical requirement for sensor simulators to effectively regress driving actions. Although contemporary 3D Gaussian Splatting (3DGS) techniques achieve remarkable reconstruction quality, their direct extension through the integration of diffusion priors often introduces cumulative physical inconsistencies and compromises training efficiency. To address these limitations, we present RGE-GS, a novel expansive reconstruction framework that synergizes diffusion-based generation with reward-guided Gaussian integration. The RGE-GS framework incorporates two key innovations: First, we propose a reward network that learns to identify and prioritize consistently generated patterns prior to reconstruction phases, thereby enabling selective retention of diffusion outputs for spatial stability. Second, during the reconstruction process, we devise a differentiated training strategy that automatically adjust Gaussian optimization progress according to scene converge metrics, which achieving better convergence than baseline methods. Extensive evaluations of publicly available datasets demonstrate that RGE-GS achieves state-of-the-art performance in reconstruction quality. Our source-code will be made publicly available at https://github.com/CN-ADLab/RGE-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02363v1">LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Due to the complex and highly dynamic motions in the real world, synthesizing dynamic videos from multi-view inputs for arbitrary viewpoints is challenging. Previous works based on neural radiance field or 3D Gaussian splatting are limited to modeling fine-scale motion, greatly restricting their application. In this paper, we introduce LocalDyGS, which consists of two parts to adapt our method to both large-scale and fine-scale motion scenes: 1) We decompose a complex dynamic scene into streamlined local spaces defined by seeds, enabling global modeling by capturing motion within each local space. 2) We decouple static and dynamic features for local space motion modeling. A static feature shared across time steps captures static information, while a dynamic residual field provides time-specific features. These are combined and decoded to generate Temporal Gaussians, modeling motion within each local space. As a result, we propose a novel dynamic scene reconstruction framework to model highly dynamic real-world scenes more realistically. Our method not only demonstrates competitive performance on various fine-scale datasets compared to state-of-the-art (SOTA) methods, but also represents the first attempt to model larger and more complex highly dynamic scenes. Project page: https://wujh2001.github.io/LocalDyGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02257v1">Gbake: Baking 3D Gaussian Splats into Reflection Probes</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 SIGGRAPH 2025 Posters
    </div>
    <details class="paper-abstract">
      The growing popularity of 3D Gaussian Splatting has created the need to integrate traditional computer graphics techniques and assets in splatted environments. Since 3D Gaussian primitives encode lighting and geometry jointly as appearance, meshes are relit improperly when inserted directly in a mixture of 3D Gaussians and thus appear noticeably out of place. We introduce GBake, a specialized tool for baking reflection probes from Gaussian-splatted scenes that enables realistic reflection mapping of traditional 3D meshes in the Unity game engine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20309v6">InstantSplat: Sparse-view Gaussian Splatting in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Project Page: https://instantsplat.github.io/
    </div>
    <details class="paper-abstract">
      While neural 3D reconstruction has advanced substantially, its performance significantly degrades with sparse-view data, which limits its broader applicability, since SfM is often unreliable in sparse-view scenarios where feature matches are scarce. In this paper, we introduce InstantSplat, a novel approach for addressing sparse-view 3D scene reconstruction at lightning-fast speed. InstantSplat employs a self-supervised framework that optimizes 3D scene representation and camera poses by unprojecting 2D pixels into 3D space and aligning them using differentiable neural rendering. The optimization process is initialized with a large-scale trained geometric foundation model, which provides dense priors that yield initial points through model inference, after which we further optimize all scene parameters using photometric errors. To mitigate redundancy introduced by the prior model, we propose a co-visibility-based geometry initialization, and a Gaussian-based bundle adjustment is employed to rapidly adapt both the scene representation and camera parameters without relying on a complex adaptive density control process. Overall, InstantSplat is compatible with multiple point-based representations for view synthesis and surface reconstruction. It achieves an acceleration of over 30x in reconstruction and improves visual quality (SSIM) from 0.3755 to 0.7624 compared to traditional SfM with 3D-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12089v3">FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 accepted in CVPR 2025, project page https://fanguw.github.io/FruitNinja3D
    </div>
    <details class="paper-abstract">
      In the real world, objects reveal internal textures when sliced or cut, yet this behavior is not well-studied in 3D generation tasks today. For example, slicing a virtual 3D watermelon should reveal flesh and seeds. Given that no available dataset captures an object's full internal structure and collecting data from all slices is impractical, generative methods become the obvious approach. However, current 3D generation and inpainting methods often focus on visible appearance and overlook internal textures. To bridge this gap, we introduce FruitNinja, the first method to generate internal textures for 3D objects undergoing geometric and topological changes. Our approach produces objects via 3D Gaussian Splatting (3DGS) with both surface and interior textures synthesized, enabling real-time slicing and rendering without additional optimization. FruitNinja leverages a pre-trained diffusion model to progressively inpaint cross-sectional views and applies voxel-grid-based smoothing to achieve cohesive textures throughout the object. Our OpaqueAtom GS strategy overcomes 3DGS limitations by employing densely distributed opaque Gaussians, avoiding biases toward larger particles that destabilize training and sharp color transitions for fine-grained textures. Experimental results show that FruitNinja substantially outperforms existing approaches, showcasing unmatched visual quality in real-time rendered internal views across arbitrary geometry manipulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14825v2">GraphGSOcc: Semantic-Geometric Graph Transformer with Dynamic-Static Decoupling for 3D Gaussian Splatting-based Occupancy Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Addressing the task of 3D semantic occupancy prediction for autonomous driving, we tackle two key issues in existing 3D Gaussian Splatting (3DGS) methods: (1) unified feature aggregation neglecting semantic correlations among similar categories and across regions, (2) boundary ambiguities caused by the lack of geometric constraints in MLP iterative optimization and (3) biased issues in dynamic-static object coupling optimization. We propose the GraphGSOcc model, a novel framework that combines semantic and geometric graph Transformer and decouples dynamic-static objects optimization for 3D Gaussian Splatting-based Occupancy Prediction. We propose the Dual Gaussians Graph Attenntion, which dynamically constructs dual graph structures: a geometric graph adaptively calculating KNN search radii based on Gaussian poses, enabling large-scale Gaussians to aggregate features from broader neighborhoods while compact Gaussians focus on local geometric consistency; a semantic graph retaining top-M highly correlated nodes via cosine similarity to explicitly encode semantic relationships within and across instances. Coupled with the Multi-scale Graph Attention framework, fine-grained attention at lower layers optimizes boundary details, while coarsegrained attention at higher layers models object-level topology. On the other hand, we decouple dynamic and static objects by leveraging semantic probability distributions and design a Dynamic-Static Decoupled Gaussian Attention mechanism to optimize the prediction performance for both dynamic objects and static scenes. GraphGSOcc achieves state-ofthe-art performance on the SurroundOcc-nuScenes, Occ3D-nuScenes, OpenOcc and KITTI occupancy benchmarks. Experiments on the SurroundOcc dataset achieve an mIoU of 25.20%, reducing GPU memory to 6.8 GB, demonstrating a 1.97% mIoU improvement and 13.7% memory reduction compared to GaussianWorld.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13176v2">DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Reconstructing clean, distractor-free 3D scenes from real-world captures remains a significant challenge, particularly in highly dynamic and cluttered settings such as egocentric videos. To tackle this problem, we introduce DeGauss, a simple and robust self-supervised framework for dynamic scene reconstruction based on a decoupled dynamic-static Gaussian Splatting design. DeGauss models dynamic elements with foreground Gaussians and static content with background Gaussians, using a probabilistic mask to coordinate their composition and enable independent yet complementary optimization. DeGauss generalizes robustly across a wide range of real-world scenarios, from casual image collections to long, dynamic egocentric videos, without relying on complex heuristics or extensive supervision. Experiments on benchmarks including NeRF-on-the-go, ADT, AEA, Hot3D, and EPIC-Fields demonstrate that DeGauss consistently outperforms existing methods, establishing a strong baseline for generalizable, distractor-free 3D reconstructionin highly dynamic, interaction-rich environments. Project page: https://batfacewayne.github.io/DeGauss.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01367v1">3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23309v2">SurgTPGS: Semantic 3D Surgical Scene Understanding with Text Promptable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 MICCAI 2025. Project Page: https://lastbasket.github.io/MICCAI-2025-SurgTPGS/
    </div>
    <details class="paper-abstract">
      In contemporary surgical research and practice, accurately comprehending 3D surgical scenes with text-promptable capabilities is particularly crucial for surgical planning and real-time intra-operative guidance, where precisely identifying and interacting with surgical tools and anatomical structures is paramount. However, existing works focus on surgical vision-language model (VLM), 3D reconstruction, and segmentation separately, lacking support for real-time text-promptable 3D queries. In this paper, we present SurgTPGS, a novel text-promptable Gaussian Splatting method to fill this gap. We introduce a 3D semantics feature learning strategy incorporating the Segment Anything model and state-of-the-art vision-language models. We extract the segmented language features for 3D surgical scene reconstruction, enabling a more in-depth understanding of the complex surgical environment. We also propose semantic-aware deformation tracking to capture the seamless deformation of semantic features, providing a more precise reconstruction for both texture and semantic features. Furthermore, we present semantic region-aware optimization, which utilizes regional-based semantic information to supervise the training, particularly promoting the reconstruction quality and semantic smoothness. We conduct comprehensive experiments on two real-world surgical datasets to demonstrate the superiority of SurgTPGS over state-of-the-art methods, highlighting its potential to revolutionize surgical practices. SurgTPGS paves the way for developing next-generation intelligent surgical systems by enhancing surgical precision and safety. Our code is available at: https://github.com/lastbasket/SurgTPGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22099v2">BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Accepted at ICCV 2025, Project Page: https://github.com/fudan-zvg/BezierGS
    </div>
    <details class="paper-abstract">
      The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose B\'ezier curve Gaussian splatting (B\'ezierGS), which represents the motion trajectories of dynamic objects using learnable B\'ezier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that B\'ezierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07007v3">Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Accepted by IJCAI 2025 Survey Track
    </div>
    <details class="paper-abstract">
      Recent advancements in AI-generated content have significantly improved the realism of 3D and 4D generation. However, most existing methods prioritize appearance consistency while neglecting underlying physical principles, leading to artifacts such as unrealistic deformations, unstable dynamics, and implausible objects interactions. Incorporating physics priors into generative models has become a crucial research direction to enhance structural integrity and motion realism. This survey provides a review of physics-aware generative methods, systematically analyzing how physical constraints are integrated into 3D and 4D generation. First, we examine recent works in incorporating physical priors into static and dynamic 3D generation, categorizing methods based on representation types, including vision-based, NeRF-based, and Gaussian Splatting-based approaches. Second, we explore emerging techniques in 4D generation, focusing on methods that model temporal dynamics with physical simulations. Finally, we conduct a comparative analysis of major methods, highlighting their strengths, limitations, and suitability for different materials and motion dynamics. By presenting an in-depth analysis of physics-grounded AIGC, this survey aims to bridge the gap between generative models and physical realism, providing insights that inspire future research in physically consistent content generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01125v1">VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present VISTA (Viewpoint-based Image selection with Semantic Task Awareness), an active exploration method for robots to plan informative trajectories that improve 3D map quality in areas most relevant for task completion. Given an open-vocabulary search instruction (e.g., "find a person"), VISTA enables a robot to explore its environment to search for the object of interest, while simultaneously building a real-time semantic 3D Gaussian Splatting reconstruction of the scene. The robot navigates its environment by planning receding-horizon trajectories that prioritize semantic similarity to the query and exploration of unseen regions of the environment. To evaluate trajectories, VISTA introduces a novel, efficient viewpoint-semantic coverage metric that quantifies both the geometric view diversity and task relevance in the 3D scene. On static datasets, our coverage metric outperforms state-of-the-art baselines, FisherRF and Bayes' Rays, in computation speed and reconstruction quality. In quadrotor hardware experiments, VISTA achieves 6x higher success rates in challenging maps, compared to baseline methods, while matching baseline performance in less challenging maps. Lastly, we show that VISTA is platform-agnostic by deploying it on a quadrotor drone and a Spot quadruped robot. Open-source code will be released upon acceptance of the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01110v1">A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks -- a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously. We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU -- without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to support real-time streaming and rendering. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes -- from broad aerial views to fine-grained ground-level details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00916v1">Masks make discriminative models great again!</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      We present Image2GS, a novel approach that addresses the challenging problem of reconstructing photorealistic 3D scenes from a single image by focusing specifically on the image-to-3D lifting component of the reconstruction process. By decoupling the lifting problem (converting an image to a 3D model representing what is visible) from the completion problem (hallucinating content not present in the input), we create a more deterministic task suitable for discriminative models. Our method employs visibility masks derived from optimized 3D Gaussian splats to exclude areas not visible from the source view during training. This masked training strategy significantly improves reconstruction quality in visible regions compared to strong baselines. Notably, despite being trained only on masked regions, Image2GS remains competitive with state-of-the-art discriminative models trained on full target images when evaluated on complete scenes. Our findings highlight the fundamental struggle discriminative models face when fitting unseen regions and demonstrate the advantages of addressing image-to-3D lifting as a distinct problem with specialized techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00886v1">GaussianVLM: Scene-centric 3D Vision-Language Models using Language-aligned Gaussian Splats for Embodied Reasoning and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      As multimodal language models advance, their application to 3D scene understanding is a fast-growing frontier, driving the development of 3D Vision-Language Models (VLMs). Current methods show strong dependence on object detectors, introducing processing bottlenecks and limitations in taxonomic flexibility. To address these limitations, we propose a scene-centric 3D VLM for 3D Gaussian splat scenes that employs language- and task-aware scene representations. Our approach directly embeds rich linguistic features into the 3D scene representation by associating language with each Gaussian primitive, achieving early modality alignment. To process the resulting dense representations, we introduce a dual sparsifier that distills them into compact, task-relevant tokens via task-guided and location-guided pathways, producing sparse, task-aware global and local scene tokens. Notably, we present the first Gaussian splatting-based VLM, leveraging photorealistic 3D representations derived from standard RGB images, demonstrating strong generalization: it improves performance of prior 3D VLM five folds, in out-of-the-domain settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00554v1">LOD-GS: Level-of-Detail-Sensitive 3D Gaussian Splatting for Detail Conserved Anti-Aliasing</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Despite the advancements in quality and efficiency achieved by 3D Gaussian Splatting (3DGS) in 3D scene rendering, aliasing artifacts remain a persistent challenge. Existing approaches primarily rely on low-pass filtering to mitigate aliasing. However, these methods are not sensitive to the sampling rate, often resulting in under-filtering and over-smoothing renderings. To address this limitation, we propose LOD-GS, a Level-of-Detail-sensitive filtering framework for Gaussian Splatting, which dynamically predicts the optimal filtering strength for each 3D Gaussian primitive. Specifically, we introduce a set of basis functions to each Gaussian, which take the sampling rate as input to model appearance variations, enabling sampling-rate-sensitive filtering. These basis function parameters are jointly optimized with the 3D Gaussian in an end-to-end manner. The sampling rate is influenced by both focal length and camera distance. However, existing methods and datasets rely solely on down-sampling to simulate focal length changes for anti-aliasing evaluation, overlooking the impact of camera distance. To enable a more comprehensive assessment, we introduce a new synthetic dataset featuring objects rendered at varying camera distances. Extensive experiments on both public datasets and our newly collected dataset demonstrate that our method achieves SOTA rendering quality while effectively eliminating aliasing. The code and dataset have been open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00363v1">GDGS: 3D Gaussian Splatting Via Geometry-Guided Initialization And Dynamic Density Control</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      We propose a method to enhance 3D Gaussian Splatting (3DGS)~\cite{Kerbl2023}, addressing challenges in initialization, optimization, and density control. Gaussian Splatting is an alternative for rendering realistic images while supporting real-time performance, and it has gained popularity due to its explicit 3D Gaussian representation. However, 3DGS heavily depends on accurate initialization and faces difficulties in optimizing unstructured Gaussian distributions into ordered surfaces, with limited adaptive density control mechanism proposed so far. Our first key contribution is a geometry-guided initialization to predict Gaussian parameters, ensuring precise placement and faster convergence. We then introduce a surface-aligned optimization strategy to refine Gaussian placement, improving geometric accuracy and aligning with the surface normals of the scene. Finally, we present a dynamic adaptive density control mechanism that adjusts Gaussian density based on regional complexity, for visual fidelity. These innovations enable our method to achieve high-fidelity real-time rendering and significant improvements in visual quality, even in complex scenes. Our method demonstrates comparable or superior results to state-of-the-art methods, rendering high-fidelity images in real time.
    </details>
</div>
