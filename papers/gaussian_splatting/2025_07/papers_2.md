# gaussian splatting - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

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
