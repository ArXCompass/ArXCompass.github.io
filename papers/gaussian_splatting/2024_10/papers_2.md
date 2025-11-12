# gaussian splatting - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14959v3">EvGGS: A Collaborative Learning Framework for Event-based Generalizable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Event cameras offer promising advantages such as high dynamic range and low latency, making them well-suited for challenging lighting conditions and fast-moving scenarios. However, reconstructing 3D scenes from raw event streams is difficult because event data is sparse and does not carry absolute color information. To release its potential in 3D reconstruction, we propose the first event-based generalizable 3D reconstruction framework, called EvGGS, which reconstructs scenes as 3D Gaussians from only event input in a feedforward manner and can generalize to unseen cases without any retraining. This framework includes a depth estimation module, an intensity reconstruction module, and a Gaussian regression module. These submodules connect in a cascading manner, and we collaboratively train them with a designed joint loss to make them mutually promote. To facilitate related studies, we build a novel event-based 3D dataset with various material objects and calibrated labels of grayscale images, depth maps, camera poses, and silhouettes. Experiments show models that have jointly trained significantly outperform those trained individually. Our approach performs better than all baselines in reconstruction quality, and depth/intensity predictions with satisfactory rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06231v2">RelitLRM: Generative Relightable Radiance for Large Reconstruction Models</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 webpage: https://relit-lrm.github.io/
    </div>
    <details class="paper-abstract">
      We propose RelitLRM, a Large Reconstruction Model (LRM) for generating high-quality Gaussian splatting representations of 3D objects under novel illuminations from sparse (4-8) posed images captured under unknown static lighting. Unlike prior inverse rendering methods requiring dense captures and slow optimization, often causing artifacts like incorrect highlights or shadow baking, RelitLRM adopts a feed-forward transformer-based model with a novel combination of a geometry reconstructor and a relightable appearance generator based on diffusion. The model is trained end-to-end on synthetic multi-view renderings of objects under varying known illuminations. This architecture design enables to effectively decompose geometry and appearance, resolve the ambiguity between material and lighting, and capture the multi-modal distribution of shadows and specularity in the relit appearance. We show our sparse-view feed-forward RelitLRM offers competitive relighting results to state-of-the-art dense-view optimization-based baselines while being significantly faster. Our project page is available at: https://relit-lrm.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07577v1">3D Vision-Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 main paper + supplementary material
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D reconstruction methods and vision-language models have propelled the development of multi-modal 3D scene understanding, which has vital applications in robotics, autonomous driving, and virtual/augmented reality. However, current multi-modal scene understanding approaches have naively embedded semantic representations into 3D reconstruction methods without striking a balance between visual and language modalities, which leads to unsatisfying semantic rasterization of translucent or reflective objects, as well as over-fitting on color modality. To alleviate these limitations, we propose a solution that adequately handles the distinct visual and semantic modalities, i.e., a 3D vision-language Gaussian splatting model for scene understanding, to put emphasis on the representation learning of language modality. We propose a novel cross-modal rasterizer, using modality fusion along with a smoothed semantic indicator for enhancing semantic rasterization. We also employ a camera-view blending technique to improve semantic consistency between existing and synthesized views, thereby effectively mitigating over-fitting. Extensive experiments demonstrate that our method achieves state-of-the-art performance in open-vocabulary semantic segmentation, surpassing existing methods by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07090v3">3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Project page: https://gaussiantracer.github.io/. Published at SIGGRAPH Asia 2024
    </div>
    <details class="paper-abstract">
      Particle-based representations of radiance fields such as 3D Gaussian Splatting have found great success for reconstructing and re-rendering of complex scenes. Most existing methods render particles via rasterization, projecting them to screen space tiles for processing in a sorted order. This work instead considers ray tracing the particles, building a bounding volume hierarchy and casting a ray for each pixel using high-performance GPU ray tracing hardware. To efficiently handle large numbers of semi-transparent particles, we describe a specialized rendering algorithm which encapsulates particles with bounding meshes to leverage fast ray-triangle intersections, and shades batches of intersections in depth-order. The benefits of ray tracing are well-known in computer graphics: processing incoherent rays for secondary lighting effects such as shadows and reflections, rendering from highly-distorted cameras common in robotics, stochastically sampling rays, and more. With our renderer, this flexibility comes at little cost compared to rasterization. Experiments demonstrate the speed and accuracy of our approach, as well as several applications in computer graphics and vision. We further propose related improvements to the basic Gaussian representation, including a simple use of generalized kernel functions which significantly reduces particle hit counts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.00525v3">StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 SIGGRAPH 2024 (Journal Track); Project Page: https://r4dl.github.io/StopThePop/
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a prominent model for constructing 3D representations from images across diverse domains. However, the efficiency of the 3D Gaussian Splatting rendering pipeline relies on several simplifications. Notably, reducing Gaussian to 2D splats with a single view-space depth introduces popping and blending artifacts during view rotation. Addressing this issue requires accurate per-pixel depth computation, yet a full per-pixel sort proves excessively costly compared to a global sort operation. In this paper, we present a novel hierarchical rasterization approach that systematically resorts and culls splats with minimal processing overhead. Our software rasterizer effectively eliminates popping artifacts and view inconsistencies, as demonstrated through both quantitative and qualitative measurements. Simultaneously, our method mitigates the potential for cheating view-dependent effects with popping, ensuring a more authentic representation. Despite the elimination of cheating, our approach achieves comparable quantitative results for test images, while increasing the consistency for novel view synthesis in motion. Due to its design, our hierarchical approach is only 4% slower on average than the original Gaussian Splatting. Notably, enforcing consistency enables a reduction in the number of Gaussians by approximately half with nearly identical quality and view-consistency. Consequently, rendering performance is nearly doubled, making our approach 1.6x faster than the original Gaussian Splatting, with a 50% reduction in memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05008v3">FlowDreamer: Exploring High Fidelity Text-to-3D Generation via Rectified Flow</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Tech Report
    </div>
    <details class="paper-abstract">
      Recent advances in text-to-3D generation have made significant progress. In particular, with the pretrained diffusion models, existing methods predominantly use Score Distillation Sampling (SDS) to train 3D models such as Neural RaRecent advances in text-to-3D generation have made significant progress. In particular, with the pretrained diffusion models, existing methods predominantly use Score Distillation Sampling (SDS) to train 3D models such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3D GS). However, a hurdle is that they often encounter difficulties with over-smoothing textures and over-saturating colors. The rectified flow model -- which utilizes a simple ordinary differential equation (ODE) to represent a straight trajectory -- shows promise as an alternative prior to text-to-3D generation. It learns a time-independent vector field, thereby reducing the ambiguity in 3D model update gradients that are calculated using time-dependent scores in the SDS framework. In light of this, we first develop a mathematical analysis to seamlessly integrate SDS with rectified flow model, paving the way for our initial framework known as Vector Field Distillation Sampling (VFDS). However, empirical findings indicate that VFDS still results in over-smoothing outcomes. Therefore, we analyze the grounding reasons for such a failure from the perspective of ODE trajectories. On top, we propose a novel framework, named FlowDreamer, which yields high fidelity results with richer textual details and faster convergence. The key insight is to leverage the coupling and reversible properties of the rectified flow model to search for the corresponding noise, rather than using randomly sampled noise as in VFDS. Accordingly, we introduce a novel Unique Couple Matching (UCM) loss, which guides the 3D model to optimize along the same trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01003v2">Free-DyGS: Camera-Pose-Free Scene Reconstruction based on Gaussian Splatting for Dynamic Surgical Videos</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Reconstructing endoscopic videos is crucial for high-fidelity visualization and the efficiency of surgical operations. Despite the importance, existing 3D reconstruction methods encounter several challenges, including stringent demands for accuracy, imprecise camera positioning, intricate dynamic scenes, and the necessity for rapid reconstruction. Addressing these issues, this paper presents the first camera-pose-free scene reconstruction framework, Free-DyGS, tailored for dynamic surgical videos, leveraging 3D Gaussian splatting technology. Our approach employs a frame-by-frame reconstruction strategy and is delineated into four distinct phases: Scene Initialization, Joint Learning, Scene Expansion, and Retrospective Learning. We introduce a Generalizable Gaussians Parameterization module within the Scene Initialization and Expansion phases to proficiently generate Gaussian attributes for each pixel from the RGBD frames. The Joint Learning phase is crafted to concurrently deduce scene deformation and camera pose, facilitated by an innovative flexible deformation module. In the scene expansion stage, the Gaussian points gradually grow as the camera moves. The Retrospective Learning phase is dedicated to enhancing the precision of scene deformation through the reassessment of prior frames. The efficacy of the proposed Free-DyGS is substantiated through experiments on two datasets: the StereoMIS and Hamlyn datasets. The experimental outcomes underscore that Free-DyGS surpasses conventional baseline models in both rendering fidelity and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06475v1">3D Representation Methods: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Preliminary Draft
    </div>
    <details class="paper-abstract">
      The field of 3D representation has experienced significant advancements, driven by the increasing demand for high-fidelity 3D models in various applications such as computer graphics, virtual reality, and autonomous systems. This review examines the development and current state of 3D representation methods, highlighting their research trajectories, innovations, strength and weakness. Key techniques such as Voxel Grid, Point Cloud, Mesh, Signed Distance Function (SDF), Neural Radiance Field (NeRF), 3D Gaussian Splatting, Tri-Plane, and Deep Marching Tetrahedra (DMTet) are reviewed. The review also introduces essential datasets that have been pivotal in advancing the field, highlighting their characteristics and impact on research progress. Finally, we explore potential research directions that hold promise for further expanding the capabilities and applications of 3D representation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17624v2">HGS-Planner: Hierarchical Planning Framework for Active Scene Reconstruction Using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      In complex missions such as search and rescue,robots must make intelligent decisions in unknown environments, relying on their ability to perceive and understand their surroundings. High-quality and real-time reconstruction enhances situational awareness and is crucial for intelligent robotics. Traditional methods often struggle with poor scene representation or are too slow for real-time use. Inspired by the efficacy of 3D Gaussian Splatting (3DGS), we propose a hierarchical planning framework for fast and high-fidelity active reconstruction. Our method evaluates completion and quality gain to adaptively guide reconstruction, integrating global and local planning for efficiency. Experiments in simulated and real-world environments show our approach outperforms existing real-time methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01053v2">HAHA: Highly Articulated Gaussian Human Avatars with Textured Mesh Prior</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      We present HAHA - a novel approach for animatable human avatar generation from monocular input videos. The proposed method relies on learning the trade-off between the use of Gaussian splatting and a textured mesh for efficient and high fidelity rendering. We demonstrate its efficiency to animate and render full-body human avatars controlled via the SMPL-X parametric model. Our model learns to apply Gaussian splatting only in areas of the SMPL-X mesh where it is necessary, like hair and out-of-mesh clothing. This results in a minimal number of Gaussians being used to represent the full avatar, and reduced rendering artifacts. This allows us to handle the animation of small body parts such as fingers that are traditionally disregarded. We demonstrate the effectiveness of our approach on two open datasets: SnapshotPeople and X-Humans. Our method demonstrates on par reconstruction quality to the state-of-the-art on SnapshotPeople, while using less than a third of Gaussians. HAHA outperforms previous state-of-the-art on novel poses from X-Humans both quantitatively and qualitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00525v3">StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 SIGGRAPH 2024 (Journal Track); Project Page: https://r4dl.github.io/StopThePop/
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a prominent model for constructing 3D representations from images across diverse domains. However, the efficiency of the 3D Gaussian Splatting rendering pipeline relies on several simplifications. Notably, reducing Gaussian to 2D splats with a single view-space depth introduces popping and blending artifacts during view rotation. Addressing this issue requires accurate per-pixel depth computation, yet a full per-pixel sort proves excessively costly compared to a global sort operation. In this paper, we present a novel hierarchical rasterization approach that systematically resorts and culls splats with minimal processing overhead. Our software rasterizer effectively eliminates popping artifacts and view inconsistencies, as demonstrated through both quantitative and qualitative measurements. Simultaneously, our method mitigates the potential for cheating view-dependent effects with popping, ensuring a more authentic representation. Despite the elimination of cheating, our approach achieves comparable quantitative results for test images, while increasing the consistency for novel view synthesis in motion. Due to its design, our hierarchical approach is only 4% slower on average than the original Gaussian Splatting. Notably, enforcing consistency enables a reduction in the number of Gaussians by approximately half with nearly identical quality and view-consistency. Consequently, rendering performance is nearly doubled, making our approach 1.6x faster than the original Gaussian Splatting, with a 50% reduction in memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12518v2">Hi-SLAM: Scaling-up Semantics in SLAM with a Hierarchically Categorical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We propose Hi-SLAM, a semantic 3D Gaussian Splatting SLAM method featuring a novel hierarchical categorical representation, which enables accurate global 3D semantic mapping, scaling-up capability, and explicit semantic label prediction in the 3D world. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making it particularly challenging and costly for scene understanding. To address this problem, we introduce a novel hierarchical representation that encodes semantic information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs). We further introduce a novel semantic loss designed to optimize hierarchical semantic information through both inter-level and cross-level optimization. Furthermore, we enhance the whole SLAM system, resulting in improved tracking and mapping performance. Our Hi-SLAM outperforms existing dense SLAM methods in both mapping and tracking accuracy, while achieving a 2x operation speed-up. Additionally, it exhibits competitive performance in rendering semantic segmentation in small synthetic scenes, with significantly reduced storage and training time requirements. Rendering FPS impressively reaches 2,000 with semantic information and 3,000 without it. Most notably, it showcases the capability of handling the complex real-world scene with more than 500 semantic classes, highlighting its valuable scaling-up capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06756v1">DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recent advancements in 2D/3D generative techniques have facilitated the generation of dynamic 3D objects from monocular videos. Previous methods mainly rely on the implicit neural radiance fields (NeRF) or explicit Gaussian Splatting as the underlying representation, and struggle to achieve satisfactory spatial-temporal consistency and surface appearance. Drawing inspiration from modern 3D animation pipelines, we introduce DreamMesh4D, a novel framework combining mesh representation with geometric skinning technique to generate high-quality 4D object from a monocular video. Instead of utilizing classical texture map for appearance, we bind Gaussian splats to triangle face of mesh for differentiable optimization of both the texture and mesh vertices. In particular, DreamMesh4D begins with a coarse mesh obtained through an image-to-3D generation procedure. Sparse points are then uniformly sampled across the mesh surface, and are used to build a deformation graph to drive the motion of the 3D object for the sake of computational efficiency and providing additional constraint. For each step, transformations of sparse control points are predicted using a deformation network, and the mesh vertices as well as the surface Gaussians are deformed via a novel geometric skinning algorithm, which is a hybrid approach combining LBS (linear blending skinning) and DQS (dual-quaternion skinning), mitigating drawbacks associated with both approaches. The static surface Gaussians and mesh vertices as well as the deformation network are learned via reference view photometric loss, score distillation loss as well as other regularizers in a two-stage manner. Extensive experiments demonstrate superior performance of our method. Furthermore, our method is compatible with modern graphic pipelines, showcasing its potential in the 3D gaming and film industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06245v1">HiSplat: Hierarchical 3D Gaussian Splatting for Generalizable Sparse-View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes from multiple viewpoints is a fundamental task in stereo vision. Recently, advances in generalizable 3D Gaussian Splatting have enabled high-quality novel view synthesis for unseen scenes from sparse input views by feed-forward predicting per-pixel Gaussian parameters without extra optimization. However, existing methods typically generate single-scale 3D Gaussians, which lack representation of both large-scale structure and texture details, resulting in mislocation and artefacts. In this paper, we propose a novel framework, HiSplat, which introduces a hierarchical manner in generalizable 3D Gaussian Splatting to construct hierarchical 3D Gaussians via a coarse-to-fine strategy. Specifically, HiSplat generates large coarse-grained Gaussians to capture large-scale structures, followed by fine-grained Gaussians to enhance delicate texture details. To promote inter-scale interactions, we propose an Error Aware Module for Gaussian compensation and a Modulating Fusion Module for Gaussian repair. Our method achieves joint optimization of hierarchical representations, allowing for novel view synthesis using only two-view reference images. Comprehensive experiments on various datasets demonstrate that HiSplat significantly enhances reconstruction quality and cross-dataset generalization compared to prior single-scale methods. The corresponding ablation study and analysis of different-scale 3D Gaussians reveal the mechanism behind the effectiveness. Project website: https://open3dvlab.github.io/HiSplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06165v1">GSLoc: Visual Localization with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      We present GSLoc: a new visual localization method that performs dense camera alignment using 3D Gaussian Splatting as a map representation of the scene. GSLoc backpropagates pose gradients over the rendering pipeline to align the rendered and target images, while it adopts a coarse-to-fine strategy by utilizing blurring kernels to mitigate the non-convexity of the problem and improve the convergence. The results show that our approach succeeds at visual localization in challenging conditions of relatively small overlap between initial and target frames inside textureless environments when state-of-the-art neural sparse methods provide inferior results. Using the byproduct of realistic rendering from the 3DGS map representation, we show how to enhance localization results by mixing a set of observed and virtual reference keyframes when solving the image retrieval problem. We evaluate our method both on synthetic and real-world data, discussing its advantages and application potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06014v1">SplaTraj: Camera Trajectory Generation with Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      Many recent developments for robots to represent environments have focused on photorealistic reconstructions. This paper particularly focuses on generating sequences of images from the photorealistic Gaussian Splatting models, that match instructions that are given by user-inputted language. We contribute a novel framework, SplaTraj, which formulates the generation of images within photorealistic environment representations as a continuous-time trajectory optimization problem. Costs are designed so that a camera following the trajectory poses will smoothly traverse through the environment and render the specified spatial information in a photogenic manner. This is achieved by querying a photorealistic representation with language embedding to isolate regions that correspond to the user-specified inputs. These regions are then projected to the camera's view as it moves over time and a cost is constructed. We can then apply gradient-based optimization and differentiate through the rendering to optimize the trajectory for the defined cost. The resulting trajectory moves to photogenically view each of the specified objects. We empirically evaluate our approach on a suite of environments and instructions, and demonstrate the quality of generated image sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05772v1">Comparative Analysis of Novel View Synthesis and Photogrammetry for 3D Forest Stand Reconstruction and extraction of individual tree parameters</a></div>
    <div class="paper-meta">
      📅 2024-10-08
      | 💬 31page,15figures
    </div>
    <details class="paper-abstract">
      Accurate and efficient 3D reconstruction of trees is crucial for forest resource assessments and management. Close-Range Photogrammetry (CRP) is commonly used for reconstructing forest scenes but faces challenges like low efficiency and poor quality. Recently, Novel View Synthesis (NVS) technologies, including Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have shown promise for 3D plant reconstruction with limited images. However, existing research mainly focuses on small plants in orchards or individual trees, leaving uncertainty regarding their application in larger, complex forest stands. In this study, we collected sequential images of forest plots with varying complexity and performed dense reconstruction using NeRF and 3DGS. The resulting point clouds were compared with those from photogrammetry and laser scanning. Results indicate that NVS methods significantly enhance reconstruction efficiency. Photogrammetry struggles with complex stands, leading to point clouds with excessive canopy noise and incorrectly reconstructed trees, such as duplicated trunks. NeRF, while better for canopy regions, may produce errors in ground areas with limited views. The 3DGS method generates sparser point clouds, particularly in trunk areas, affecting diameter at breast height (DBH) accuracy. All three methods can extract tree height information, with NeRF yielding the highest accuracy; however, photogrammetry remains superior for DBH accuracy. These findings suggest that NVS methods have significant potential for 3D reconstruction of forest stands, offering valuable support for complex forest resource inventory and visualization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.04646v1">Mode-GS: Monocular Depth Guided Anchored 3D Gaussian Splatting for Robust Ground-View Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-08
    </div>
    <details class="paper-abstract">
      We present a novel-view rendering algorithm, Mode-GS, for ground-robot trajectory datasets. Our approach is based on using anchored Gaussian splats, which are designed to overcome the limitations of existing 3D Gaussian splatting algorithms. Prior neural rendering methods suffer from severe splat drift due to scene complexity and insufficient multi-view observation, and can fail to fix splats on the true geometry in ground-robot datasets. Our method integrates pixel-aligned anchors from monocular depths and generates Gaussian splats around these anchors using residual-form Gaussian decoders. To address the inherent scale ambiguity of monocular depth, we parameterize anchors with per-view depth-scales and employ scale-consistent depth loss for online scale calibration. Our method results in improved rendering performance, based on PSNR, SSIM, and LPIPS metrics, in ground scenes with free trajectory patterns, and achieves state-of-the-art rendering performance on the R3LIVE odometry dataset and the Tanks and Temples dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05259v1">GS-VTON: Controllable 3D Virtual Try-on with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 21 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Diffusion-based 2D virtual try-on (VTON) techniques have recently demonstrated strong performance, while the development of 3D VTON has largely lagged behind. Despite recent advances in text-guided 3D scene editing, integrating 2D VTON into these pipelines to achieve vivid 3D VTON remains challenging. The reasons are twofold. First, text prompts cannot provide sufficient details in describing clothing. Second, 2D VTON results generated from different viewpoints of the same 3D scene lack coherence and spatial relationships, hence frequently leading to appearance inconsistencies and geometric distortions. To resolve these problems, we introduce an image-prompted 3D VTON method (dubbed GS-VTON) which, by leveraging 3D Gaussian Splatting (3DGS) as the 3D representation, enables the transfer of pre-trained knowledge from 2D VTON models to 3D while improving cross-view consistency. (1) Specifically, we propose a personalized diffusion model that utilizes low-rank adaptation (LoRA) fine-tuning to incorporate personalized information into pre-trained 2D VTON models. To achieve effective LoRA training, we introduce a reference-driven image editing approach that enables the simultaneous editing of multi-view images while ensuring consistency. (2) Furthermore, we propose a persona-aware 3DGS editing framework to facilitate effective editing while maintaining consistent cross-view appearance and high-quality 3D geometry. (3) Additionally, we have established a new 3D VTON benchmark, 3D-VTONBench, which facilitates comprehensive qualitative and quantitative 3D VTON evaluations. Through extensive experiments and comparative analyses with existing methods, the proposed \OM has demonstrated superior fidelity and advanced editing capabilities, affirming its effectiveness for 3D VTON.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05111v1">LiDAR-GS:Real-time LiDAR Re-Simulation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      LiDAR simulation plays a crucial role in closed-loop simulation for autonomous driving. Although recent advancements, such as the use of reconstructed mesh and Neural Radiance Fields (NeRF), have made progress in simulating the physical properties of LiDAR, these methods have struggled to achieve satisfactory frame rates and rendering quality. To address these limitations, we present LiDAR-GS, the first LiDAR Gaussian Splatting method, for real-time high-fidelity re-simulation of LiDAR sensor scans in public urban road scenes. The vanilla Gaussian Splatting, designed for camera models, cannot be directly applied to LiDAR re-simulation. To bridge the gap between passive camera and active LiDAR, our LiDAR-GS designs a differentiable laser beam splatting, grounded in the LiDAR range view model. This innovation allows for precise surface splatting by projecting lasers onto micro cross-sections, effectively eliminating artifacts associated with local affine approximations. Additionally, LiDAR-GS leverages Neural Gaussian Fields, which further integrate view-dependent clues, to represent key LiDAR properties that are influenced by the incident angle and external factors. Combining these practices with some essential adaptations, e.g., dynamic instances decomposition, our approach succeeds in simultaneously re-simulating depth, intensity, and ray-drop channels, achieving state-of-the-art results in both rendering frame rate and quality on publically available large scene datasets. Our source code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05097v1">DreamSat: Towards a General 3D Model for Novel View Synthesis of Space Objects</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Presented at the 75th International Astronautical Congress, October 2024, Milan, Italy
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) enables to generate new images of a scene or convert a set of 2D images into a comprehensive 3D model. In the context of Space Domain Awareness, since space is becoming increasingly congested, NVS can accurately map space objects and debris, improving the safety and efficiency of space operations. Similarly, in Rendezvous and Proximity Operations missions, 3D models can provide details about a target object's shape, size, and orientation, allowing for better planning and prediction of the target's behavior. In this work, we explore the generalization abilities of these reconstruction techniques, aiming to avoid the necessity of retraining for each new scene, by presenting a novel approach to 3D spacecraft reconstruction from single-view images, DreamSat, by fine-tuning the Zero123 XL, a state-of-the-art single-view reconstruction model, on a high-quality dataset of 190 high-quality spacecraft models and integrating it into the DreamGaussian framework. We demonstrate consistent improvements in reconstruction quality across multiple metrics, including Contrastive Language-Image Pretraining (CLIP) score (+0.33%), Peak Signal-to-Noise Ratio (PSNR) (+2.53%), Structural Similarity Index (SSIM) (+2.38%), and Learned Perceptual Image Patch Similarity (LPIPS) (+0.16%) on a test set of 30 previously unseen spacecraft images. Our method addresses the lack of domain-specific 3D reconstruction tools in the space industry by leveraging state-of-the-art diffusion models and 3D Gaussian splatting techniques. This approach maintains the efficiency of the DreamGaussian framework while enhancing the accuracy and detail of spacecraft reconstructions. The code for this work can be accessed on GitHub (https://github.com/ARCLab-MIT/space-nvs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05044v1">PhotoReg: Photometrically Registering 3D Gaussian Splatting Models</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Building accurate representations of the environment is critical for intelligent robots to make decisions during deployment. Advances in photorealistic environment models have enabled robots to develop hyper-realistic reconstructions, which can be used to generate images that are intuitive for human inspection. In particular, the recently introduced \ac{3DGS}, which describes the scene with up to millions of primitive ellipsoids, can be rendered in real time. \ac{3DGS} has rapidly gained prominence. However, a critical unsolved problem persists: how can we fuse multiple \ac{3DGS} into a single coherent model? Solving this problem will enable robot teams to jointly build \ac{3DGS} models of their surroundings. A key insight of this work is to leverage the {duality} between photorealistic reconstructions, which render realistic 2D images from 3D structure, and \emph{3D foundation models}, which predict 3D structure from image pairs. To this end, we develop PhotoReg, a framework to register multiple photorealistic \ac{3DGS} models with 3D foundation models. As \ac{3DGS} models are generally built from monocular camera images, they have \emph{arbitrary scale}. To resolve this, PhotoReg actively enforces scale consistency among the different \ac{3DGS} models by considering depth estimates within these models. Then, the alignment is iteratively refined with fine-grained photometric losses to produce high-quality fused \ac{3DGS} models. We rigorously evaluate PhotoReg on both standard benchmark datasets and our custom-collected datasets, including with two quadruped robots. The code is released at \url{ziweny11.github.io/photoreg}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15118v2">GS-Hider: Hiding Messages into 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-07
      | 💬 Accepted by NeurIPS 2024, 3DGS steganography
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has already become the emerging research focus in the fields of 3D scene reconstruction and novel view synthesis. Given that training a 3DGS requires a significant amount of time and computational cost, it is crucial to protect the copyright, integrity, and privacy of such 3D assets. Steganography, as a crucial technique for encrypted transmission and copyright protection, has been extensively studied. However, it still lacks profound exploration targeted at 3DGS. Unlike its predecessor NeRF, 3DGS possesses two distinct features: 1) explicit 3D representation; and 2) real-time rendering speeds. These characteristics result in the 3DGS point cloud files being public and transparent, with each Gaussian point having a clear physical significance. Therefore, ensuring the security and fidelity of the original 3D scene while embedding information into the 3DGS point cloud files is an extremely challenging task. To solve the above-mentioned issue, we first propose a steganography framework for 3DGS, dubbed GS-Hider, which can embed 3D scenes and images into original GS point clouds in an invisible manner and accurately extract the hidden messages. Specifically, we design a coupled secured feature attribute to replace the original 3DGS's spherical harmonics coefficients and then use a scene decoder and a message decoder to disentangle the original RGB scene and the hidden message. Extensive experiments demonstrated that the proposed GS-Hider can effectively conceal multimodal messages without compromising rendering quality and possesses exceptional security, robustness, capacity, and flexibility. Our project is available at: https://xuanyuzhang21.github.io/project/gshider.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19215v2">1st Place Solution to the 8th HANDS Workshop Challenge -- ARCTIC Track: 3DGS-based Bimanual Category-agnostic Interaction Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      This report describes our 1st place solution to the 8th HANDS workshop challenge (ARCTIC track) in conjunction with ECCV 2024. In this challenge, we address the task of bimanual category-agnostic hand-object interaction reconstruction, which aims to generate 3D reconstructions of both hands and the object from a monocular video, without relying on predefined templates. This task is particularly challenging due to the significant occlusion and dynamic contact between the hands and the object during bimanual manipulation. We worked to resolve these issues by introducing a mask loss and a 3D contact loss, respectively. Moreover, we applied 3D Gaussian Splatting (3DGS) to this task. As a result, our method achieved a value of 38.69 in the main metric, CD$_h$, on the ARCTIC test set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10161v3">SplatSim: Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-07
    </div>
    <details class="paper-abstract">
      Sim2Real transfer, particularly for manipulation policies relying on RGB images, remains a critical challenge in robotics due to the significant domain shift between synthetic and real-world visual data. In this paper, we propose SplatSim, a novel framework that leverages Gaussian Splatting as the primary rendering primitive to reduce the Sim2Real gap for RGB-based manipulation policies. By replacing traditional mesh representations with Gaussian Splats in simulators, SplatSim produces highly photorealistic synthetic data while maintaining the scalability and cost-efficiency of simulation. We demonstrate the effectiveness of our framework by training manipulation policies within SplatSim and deploying them in the real world in a zero-shot manner, achieving an average success rate of 86.25%, compared to 97.5% for policies trained on real-world data. Videos can be found on our project page: https://splatsim.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04646v1">Mode-GS: Monocular Depth Guided Anchored 3D Gaussian Splatting for Robust Ground-View Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      We present a novel-view rendering algorithm, Mode-GS, for ground-robot trajectory datasets. Our approach is based on using anchored Gaussian splats, which are designed to overcome the limitations of existing 3D Gaussian splatting algorithms. Prior neural rendering methods suffer from severe splat drift due to scene complexity and insufficient multi-view observation, and can fail to fix splats on the true geometry in ground-robot datasets. Our method integrates pixel-aligned anchors from monocular depths and generates Gaussian splats around these anchors using residual-form Gaussian decoders. To address the inherent scale ambiguity of monocular depth, we parameterize anchors with per-view depth-scales and employ scale-consistent depth loss for online scale calibration. Our method results in improved rendering performance, based on PSNR, SSIM, and LPIPS metrics, in ground scenes with free trajectory patterns, and achieves state-of-the-art rendering performance on the R3LIVE odometry dataset and the Tanks and Temples dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13036v2">S4D: Streaming 4D Real-World Reconstruction with Gaussians and 3D Control Points</a></div>
    <div class="paper-meta">
      📅 2024-10-06
      | 💬 20 pages, 9 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction using Gaussians has recently attracted increased interest. Mainstream approaches typically employ a global deformation field to warp a 3D scene in canonical space. However, the inherent low-frequency nature of implicit neural fields often leads to ineffective representations of complex motions. Moreover, their structural rigidity can hinder adaptation to scenes with varying resolutions and durations. To address these challenges, we introduce a novel approach for streaming 4D real-world reconstruction utilizing discrete 3D control points. This method physically models local rays and establishes a motion-decoupling coordinate system. By effectively merging traditional graphics with learnable pipelines, it provides a robust and efficient local 6-degrees-of-freedom (6-DoF) motion representation. Additionally, we have developed a generalized framework that integrates our control points with Gaussians. Starting from an initial 3D reconstruction, our workflow decomposes the streaming 4D reconstruction into four independent submodules: 3D segmentation, 3D control point generation, object-wise motion manipulation, and residual compensation. Experimental results demonstrate that our method outperforms existing state-of-the-art 4D Gaussian splatting techniques on both the Neu3DV and CMU-Panoptic datasets. Notably, the optimization of our 3D control points is achievable in 100 iterations and within just 2 seconds per frame on a single NVIDIA 4070 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01535v2">GaussianBlock: Building Part-Aware Compositional and Editable 3D Scene by Primitives and Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-10-06
    </div>
    <details class="paper-abstract">
      Recently, with the development of Neural Radiance Fields and Gaussian Splatting, 3D reconstruction techniques have achieved remarkably high fidelity. However, the latent representations learnt by these methods are highly entangled and lack interpretability. In this paper, we propose a novel part-aware compositional reconstruction method, called GaussianBlock, that enables semantically coherent and disentangled representations, allowing for precise and physical editing akin to building blocks, while simultaneously maintaining high fidelity. Our GaussianBlock introduces a hybrid representation that leverages the advantages of both primitives, known for their flexible actionability and editability, and 3D Gaussians, which excel in reconstruction quality. Specifically, we achieve semantically coherent primitives through a novel attention-guided centering loss derived from 2D semantic priors, complemented by a dynamic splitting and fusion strategy. Furthermore, we utilize 3D Gaussians that hybridize with primitives to refine structural details and enhance fidelity. Additionally, a binding inheritance strategy is employed to strengthen and maintain the connection between the two. Our reconstructed scenes are evidenced to be disentangled, compositional, and compact across diverse benchmarks, enabling seamless, direct and precise editing while maintaining high quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03592v1">Variational Bayes Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting has emerged as a promising approach for modeling 3D scenes using mixtures of Gaussians. The predominant optimization method for these models relies on backpropagating gradients through a differentiable rendering pipeline, which struggles with catastrophic forgetting when dealing with continuous streams of data. To address this limitation, we propose Variational Bayes Gaussian Splatting (VBGS), a novel approach that frames training a Gaussian splat as variational inference over model parameters. By leveraging the conjugacy properties of multivariate Gaussians, we derive a closed-form variational update rule, allowing efficient updates from partial, sequential observations without the need for replay buffers. Our experiments show that VBGS not only matches state-of-the-art performance on static datasets, but also enables continual learning from sequentially streamed 2D and 3D data, drastically improving performance in this setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.02764v1">Flash-Splat: 3D Reflection Removal with Flash Cues and Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      We introduce a simple yet effective approach for separating transmitted and reflected light. Our key insight is that the powerful novel view synthesis capabilities provided by modern inverse rendering methods (e.g.,~3D Gaussian splatting) allow one to perform flash/no-flash reflection separation using unpaired measurements -- this relaxation dramatically simplifies image acquisition over conventional paired flash/no-flash reflection separation methods. Through extensive real-world experiments, we demonstrate our method, Flash-Splat, accurately reconstructs both transmitted and reflected scenes in 3D. Our method outperforms existing 3D reflection separation methods, which do not leverage illumination control, by a large margin. Our project webpage is at https://flash-splat.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02764v1">Flash-Splat: 3D Reflection Removal with Flash Cues and Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      We introduce a simple yet effective approach for separating transmitted and reflected light. Our key insight is that the powerful novel view synthesis capabilities provided by modern inverse rendering methods (e.g.,~3D Gaussian splatting) allow one to perform flash/no-flash reflection separation using unpaired measurements -- this relaxation dramatically simplifies image acquisition over conventional paired flash/no-flash reflection separation methods. Through extensive real-world experiments, we demonstrate our method, Flash-Splat, accurately reconstructs both transmitted and reflected scenes in 3D. Our method outperforms existing 3D reflection separation methods, which do not leverage illumination control, by a large margin. Our project webpage is at https://flash-splat.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02619v1">GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      We present GI-GS, a novel inverse rendering framework that leverages 3D Gaussian Splatting (3DGS) and deferred shading to achieve photo-realistic novel view synthesis and relighting. In inverse rendering, accurately modeling the shading processes of objects is essential for achieving high-fidelity results. Therefore, it is critical to incorporate global illumination to account for indirect lighting that reaches an object after multiple bounces across the scene. Previous 3DGS-based methods have attempted to model indirect lighting by characterizing indirect illumination as learnable lighting volumes or additional attributes of each Gaussian, while using baked occlusion to represent shadow effects. These methods, however, fail to accurately model the complex physical interactions between light and objects, making it impossible to construct realistic indirect illumination during relighting. To address this limitation, we propose to calculate indirect lighting using efficient path tracing with deferred shading. In our framework, we first render a G-buffer to capture the detailed geometry and material properties of the scene. Then, we perform physically-based rendering (PBR) only for direct lighting. With the G-buffer and previous rendering results, the indirect lighting can be calculated through a lightweight path tracing. Our method effectively models indirect lighting under any given lighting conditions, thereby achieving better novel view synthesis and relighting. Quantitative and qualitative results show that our GI-GS outperforms existing baselines in both rendering quality and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11682v2">SRIF: Semantic Shape Registration Empowered by Diffusion-based Image Morphing and Flow Estimation</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Accepted as a conference paper of SIGGRAPH Asia 2024
    </div>
    <details class="paper-abstract">
      In this paper, we propose SRIF, a novel Semantic shape Registration framework based on diffusion-based Image morphing and Flow estimation. More concretely, given a pair of extrinsically aligned shapes, we first render them from multi-views, and then utilize an image interpolation framework based on diffusion models to generate sequences of intermediate images between them. The images are later fed into a dynamic 3D Gaussian splatting framework, with which we reconstruct and post-process for intermediate point clouds respecting the image morphing processing. In the end, tailored for the above, we propose a novel registration module to estimate continuous normalizing flow, which deforms source shape consistently towards the target, with intermediate point clouds as weak guidance. Our key insight is to leverage large vision models (LVMs) to associate shapes and therefore obtain much richer semantic information on the relationship between shapes than the ad-hoc feature extraction and alignment. As a consequence, SRIF achieves high-quality dense correspondences on challenging shape pairs, but also delivers smooth, semantically meaningful interpolation in between. Empirical evidence justifies the effectiveness and superiority of our method as well as specific design choices. The code is released at https://github.com/rqhuang88/SRIF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02103v1">MVGS: Multi-view-regulated Gaussian Splatting for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 Project Page:https://xiaobiaodu.github.io/mvgs-project/
    </div>
    <details class="paper-abstract">
      Recent works in volume rendering, \textit{e.g.} NeRF and 3D Gaussian Splatting (3DGS), significantly advance the rendering quality and efficiency with the help of the learned implicit neural radiance field or 3D Gaussians. Rendering on top of an explicit representation, the vanilla 3DGS and its variants deliver real-time efficiency by optimizing the parametric model with single-view supervision per iteration during training which is adopted from NeRF. Consequently, certain views are overfitted, leading to unsatisfying appearance in novel-view synthesis and imprecise 3D geometries. To solve aforementioned problems, we propose a new 3DGS optimization method embodying four key novel contributions: 1) We transform the conventional single-view training paradigm into a multi-view training strategy. With our proposed multi-view regulation, 3D Gaussian attributes are further optimized without overfitting certain training views. As a general solution, we improve the overall accuracy in a variety of scenarios and different Gaussian variants. 2) Inspired by the benefit introduced by additional views, we further propose a cross-intrinsic guidance scheme, leading to a coarse-to-fine training procedure concerning different resolutions. 3) Built on top of our multi-view regulated training, we further propose a cross-ray densification strategy, densifying more Gaussian kernels in the ray-intersect regions from a selection of views. 4) By further investigating the densification strategy, we found that the effect of densification should be enhanced when certain views are distinct dramatically. As a solution, we propose a novel multi-view augmented densification strategy, where 3D Gaussians are encouraged to get densified to a sufficient number accordingly, resulting in improved reconstruction accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15870v2">Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      The recent advancements in 3D Gaussian splatting (3D-GS) have not only facilitated real-time rendering through modern GPU rasterization pipelines but have also attained state-of-the-art rendering quality. Nevertheless, despite its exceptional rendering quality and performance on standard datasets, 3D-GS frequently encounters difficulties in accurately modeling specular and anisotropic components. This issue stems from the limited ability of spherical harmonics (SH) to represent high-frequency information. To overcome this challenge, we introduce Spec-Gaussian, an approach that utilizes an anisotropic spherical Gaussian (ASG) appearance field instead of SH for modeling the view-dependent appearance of each 3D Gaussian. Additionally, we have developed a coarse-to-fine training strategy to improve learning efficiency and eliminate floaters caused by overfitting in real-world scenes. Our experimental results demonstrate that our method surpasses existing approaches in terms of rendering quality. Thanks to ASG, we have significantly improved the ability of 3D-GS to model scenes with specular and anisotropic components without increasing the number of 3D Gaussians. This improvement extends the applicability of 3D GS to handle intricate scenarios with specular and anisotropic surfaces. Project page is https://ingra14m.github.io/Spec-Gaussian-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11085v2">GSLoc: Efficient Camera Pose Refinement via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 Fixed a small bug in the first version and achieved new state-of-the-art accuracy. The project page is available at https://gsloc.active.vision
    </div>
    <details class="paper-abstract">
      We leverage 3D Gaussian Splatting (3DGS) as a scene representation and propose a novel test-time camera pose refinement framework, GSLoc. This framework enhances the localization accuracy of state-of-the-art absolute pose regression and scene coordinate regression methods. The 3DGS model renders high-quality synthetic images and depth maps to facilitate the establishment of 2D-3D correspondences. GSLoc obviates the need for training feature extractors or descriptors by operating directly on RGB images, utilizing the 3D foundation model, MASt3R, for precise 2D matching. To improve the robustness of our model in challenging outdoor environments, we incorporate an exposure-adaptive module within the 3DGS framework. Consequently, GSLoc enables efficient one-shot pose refinement given a single RGB query and a coarse initial pose estimation. Our proposed approach surpasses leading NeRF-based optimization methods in both accuracy and runtime across indoor and outdoor visual localization benchmarks, achieving new state-of-the-art accuracy on two indoor datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01647v1">3DGS-DET: Empower 3D Gaussian Splatting with Boundary Guidance and Box-Focused Sampling for 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 Code Page: https://github.com/yangcaoai/3DGS-DET
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRF) are widely used for novel-view synthesis and have been adapted for 3D Object Detection (3DOD), offering a promising approach to 3DOD through view-synthesis representation. However, NeRF faces inherent limitations: (i) limited representational capacity for 3DOD due to its implicit nature, and (ii) slow rendering speeds. Recently, 3D Gaussian Splatting (3DGS) has emerged as an explicit 3D representation that addresses these limitations. Inspired by these advantages, this paper introduces 3DGS into 3DOD for the first time, identifying two main challenges: (i) Ambiguous spatial distribution of Gaussian blobs: 3DGS primarily relies on 2D pixel-level supervision, resulting in unclear 3D spatial distribution of Gaussian blobs and poor differentiation between objects and background, which hinders 3DOD; (ii) Excessive background blobs: 2D images often include numerous background pixels, leading to densely reconstructed 3DGS with many noisy Gaussian blobs representing the background, negatively affecting detection. To tackle the challenge (i), we leverage the fact that 3DGS reconstruction is derived from 2D images, and propose an elegant and efficient solution by incorporating 2D Boundary Guidance to significantly enhance the spatial distribution of Gaussian blobs, resulting in clearer differentiation between objects and their background. To address the challenge (ii), we propose a Box-Focused Sampling strategy using 2D boxes to generate object probability distribution in 3D spaces, allowing effective probabilistic sampling in 3D to retain more object blobs and reduce noisy background blobs. Benefiting from our designs, our 3DGS-DET significantly outperforms the SOTA NeRF-based method, NeRF-Det, achieving improvements of +6.6 on mAP@0.25 and +8.1 on mAP@0.5 for the ScanNet dataset, and impressive +31.5 on mAP@0.25 for the ARKITScenes dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01614v1">Gaussian Splatting in Mirrors: Reflection-Aware Rendering via Virtual Camera Optimization</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 To be published on 2024 British Machine Vision Conference
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3D-GS) have revolutionized novel view synthesis, facilitating real-time, high-quality image rendering. However, in scenarios involving reflective surfaces, particularly mirrors, 3D-GS often misinterprets reflections as virtual spaces, resulting in blurred and inconsistent multi-view rendering within mirrors. Our paper presents a novel method aimed at obtaining high-quality multi-view consistent reflection rendering by modelling reflections as physically-based virtual cameras. We estimate mirror planes with depth and normal estimates from 3D-GS and define virtual cameras that are placed symmetrically about the mirror plane. These virtual cameras are then used to explain mirror reflections in the scene. To address imperfections in mirror plane estimates, we propose a straightforward yet effective virtual camera optimization method to enhance reflection quality. We collect a new mirror dataset including three real-world scenarios for more diverse evaluation. Experimental validation on both Mirror-Nerf and our real-world dataset demonstrate the efficacy of our approach. We achieve comparable or superior results while significantly reducing training time compared to previous state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00486v2">CaRtGS: Computational Alignment for Real-Time Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 Upon a thorough internal review, we have identified that our manuscript lacks proper citation for a critical expression within the methodology section. In this revised version, we add Taming-3DGS as a citation in the splat-wise backpropagation statement
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is pivotal in robotics, with photorealistic scene reconstruction emerging as a key challenge. To address this, we introduce Computational Alignment for Real-Time Gaussian Splatting SLAM (CaRtGS), a novel method enhancing the efficiency and quality of photorealistic scene reconstruction in real-time environments. Leveraging 3D Gaussian Splatting (3DGS), CaRtGS achieves superior rendering quality and processing speed, which is crucial for scene photorealistic reconstruction. Our approach tackles computational misalignment in Gaussian Splatting SLAM (GS-SLAM) through an adaptive strategy that optimizes training, addresses long-tail optimization, and refines densification. Experiments on Replica and TUM-RGBD datasets demonstrate CaRtGS's effectiveness in achieving high-fidelity rendering with fewer Gaussian primitives. This work propels SLAM towards real-time, photorealistic dense rendering, significantly advancing photorealistic scene representation. For the benefit of the research community, we release the code on our project website: https://dapengfeng.github.io/cartgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01521v1">MiraGe: Editable 2D Images using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-02
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) approximate discrete data through continuous functions and are commonly used for encoding 2D images. Traditional image-based INRs employ neural networks to map pixel coordinates to RGB values, capturing shapes, colors, and textures within the network's weights. Recently, GaussianImage has been proposed as an alternative, using Gaussian functions instead of neural networks to achieve comparable quality and compression. Such a solution obtains a quality and compression ratio similar to classical INR models but does not allow image modification. In contrast, our work introduces a novel method, MiraGe, which uses mirror reflections to perceive 2D images in 3D space and employs flat-controlled Gaussians for precise 2D image editing. Our approach improves the rendering quality and allows realistic image modifications, including human-inspired perception of photos in the 3D world. Thanks to modeling images in 3D space, we obtain the illusion of 3D-based modification in 2D images. We also show that our Gaussian representation can be easily combined with a physics engine to produce physics-based modification of 2D images. Consequently, MiraGe allows for better quality than the standard approach and natural modification of 2D images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01425v1">EVA-Gaussian: 3D Gaussian-based Real-time Human Novel View Synthesis under Diverse Camera Settings</a></div>
    <div class="paper-meta">
      📅 2024-10-02
    </div>
    <details class="paper-abstract">
      The feed-forward based 3D Gaussian Splatting method has demonstrated exceptional capability in real-time human novel view synthesis. However, existing approaches are restricted to dense viewpoint settings, which limits their flexibility in free-viewpoint rendering across a wide range of camera view angle discrepancies. To address this limitation, we propose a real-time pipeline named EVA-Gaussian for 3D human novel view synthesis across diverse camera settings. Specifically, we first introduce an Efficient cross-View Attention (EVA) module to accurately estimate the position of each 3D Gaussian from the source images. Then, we integrate the source images with the estimated Gaussian position map to predict the attributes and feature embeddings of the 3D Gaussians. Moreover, we employ a recurrent feature refiner to correct artifacts caused by geometric errors in position estimation and enhance visual fidelity.To further improve synthesis quality, we incorporate a powerful anchor loss function for both 3D Gaussian attributes and human face landmarks. Experimental results on the THuman2.0 and THumansit datasets showcase the superiority of our EVA-Gaussian approach in rendering quality across diverse camera settings. Project page: https://zhenliuzju.github.io/huyingdong/EVA-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01404v1">Gaussian-Det: Learning Closed-Surface Gaussians for 3D Object Detection</a></div>
    <div class="paper-meta">
      📅 2024-10-02
    </div>
    <details class="paper-abstract">
      Skins wrapping around our bodies, leathers covering over the sofa, sheet metal coating the car - it suggests that objects are enclosed by a series of continuous surfaces, which provides us with informative geometry prior for objectness deduction. In this paper, we propose Gaussian-Det which leverages Gaussian Splatting as surface representation for multi-view based 3D object detection. Unlike existing monocular or NeRF-based methods which depict the objects via discrete positional data, Gaussian-Det models the objects in a continuous manner by formulating the input Gaussians as feature descriptors on a mass of partial surfaces. Furthermore, to address the numerous outliers inherently introduced by Gaussian splatting, we accordingly devise a Closure Inferring Module (CIM) for the comprehensive surface-based objectness deduction. CIM firstly estimates the probabilistic feature residuals for partial surfaces given the underdetermined nature of Gaussian Splatting, which are then coalesced into a holistic representation on the overall surface closure of the object proposal. In this way, the surface information Gaussian-Det exploits serves as the prior on the quality and reliability of objectness and the information basis of proposal refinement. Experiments on both synthetic and real-world datasets demonstrate that Gaussian-Det outperforms various existing approaches, in terms of both average precision and recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11394v2">DreamCatalyst: Fast and High-Quality 3D Editing via Controlling Editability and Identity Preservation</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 ProjectPage: https://dream-catalyst.github.io Code: https://github.com/kaist-cvml/DreamCatalyst (Appendix included)
    </div>
    <details class="paper-abstract">
      Score distillation sampling (SDS) has emerged as an effective framework in text-driven 3D editing tasks, leveraging diffusion models for 3D consistent editing. However, existing SDS-based 3D editing methods suffer from long training times and produce low-quality results. We identify that the root cause of this performance degradation is their conflict with the sampling dynamics of diffusion models. Addressing this conflict allows us to treat SDS as a diffusion reverse process for 3D editing via sampling from data space. In contrast, existing methods naively distill the score function using diffusion models. From these insights, we propose DreamCatalyst, a novel framework that considers these sampling dynamics in the SDS framework. Specifically, we devise the optimization process of our DreamCatalyst to approximate the diffusion reverse process in editing tasks, thereby aligning with diffusion sampling dynamics. As a result, DreamCatalyst successfully reduces training time and improves editing quality. Our method offers two modes: (1) a fast mode that edits Neural Radiance Fields (NeRF) scenes approximately 23 times faster than current state-of-the-art NeRF editing methods, and (2) a high-quality mode that produces superior results about 8 times faster than these methods. Notably, our high-quality mode outperforms current state-of-the-art NeRF editing methods in terms of both speed and quality. DreamCatalyst also surpasses the state-of-the-art 3D Gaussian Splatting (3DGS) editing methods, establishing itself as an effective and model-agnostic 3D editing solution. See more extensive results on our project page: https://dream-catalyst.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12535v2">High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization</a></div>
    <div class="paper-meta">
      📅 2024-10-02
      | 💬 Accepted by IROS 2024
    </div>
    <details class="paper-abstract">
      We propose a dense RGBD SLAM system based on 3D Gaussian Splatting that provides metrically accurate pose tracking and visually realistic reconstruction. To this end, we first propose a Gaussian densification strategy based on the rendering loss to map unobserved areas and refine reobserved areas. Second, we introduce extra regularization parameters to alleviate the forgetting problem in the continuous mapping problem, where parameters tend to overfit the latest frame and result in decreasing rendering quality for previous frames. Both mapping and tracking are performed with Gaussian parameters by minimizing re-rendering loss in a differentiable way. Compared to recent neural and concurrently developed gaussian splatting RGBD SLAM baselines, our method achieves state-of-the-art results on the synthetic dataset Replica and competitive results on the real-world dataset TUM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19811v2">EgoGaussian: Dynamic Scene Understanding from Egocentric Video with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-10-02
    </div>
    <details class="paper-abstract">
      Human activities are inherently complex, often involving numerous object interactions. To better understand these activities, it is crucial to model their interactions with the environment captured through dynamic changes. The recent availability of affordable head-mounted cameras and egocentric data offers a more accessible and efficient means to understand human-object interactions in 3D environments. However, most existing methods for human activity modeling neglect the dynamic interactions with objects, resulting in only static representations. The few existing solutions often require inputs from multiple sources, including multi-camera setups, depth-sensing cameras, or kinesthetic sensors. To this end, we introduce EgoGaussian, the first method capable of simultaneously reconstructing 3D scenes and dynamically tracking 3D object motion from RGB egocentric input alone. We leverage the uniquely discrete nature of Gaussian Splatting and segment dynamic interactions from the background, with both having explicit representations. Our approach employs a clip-level online learning pipeline that leverages the dynamic nature of human activities, allowing us to reconstruct the temporal evolution of the scene in chronological order and track rigid object motion. EgoGaussian shows significant improvements in terms of both dynamic object and background reconstruction quality compared to the state-of-the-art. We also qualitatively demonstrate the high quality of the reconstructed models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00386v1">Seamless Augmented Reality Integration in Arthroscopy: A Pipeline for Articular Reconstruction and Guidance</a></div>
    <div class="paper-meta">
      📅 2024-10-01
      | 💬 8 pages, with 2 additional pages as the supplementary. Accepted by AE-CAI 2024
    </div>
    <details class="paper-abstract">
      Arthroscopy is a minimally invasive surgical procedure used to diagnose and treat joint problems. The clinical workflow of arthroscopy typically involves inserting an arthroscope into the joint through a small incision, during which surgeons navigate and operate largely by relying on their visual assessment through the arthroscope. However, the arthroscope's restricted field of view and lack of depth perception pose challenges in navigating complex articular structures and achieving surgical precision during procedures. Aiming at enhancing intraoperative awareness, we present a robust pipeline that incorporates simultaneous localization and mapping, depth estimation, and 3D Gaussian splatting to realistically reconstruct intra-articular structures solely based on monocular arthroscope video. Extending 3D reconstruction to Augmented Reality (AR) applications, our solution offers AR assistance for articular notch measurement and annotation anchoring in a human-in-the-loop manner. Compared to traditional Structure-from-Motion and Neural Radiance Field-based methods, our pipeline achieves dense 3D reconstruction and competitive rendering fidelity with explicit 3D representation in 7 minutes on average. When evaluated on four phantom datasets, our method achieves RMSE = 2.21mm reconstruction error, PSNR = 32.86 and SSIM = 0.89 on average. Because our pipeline enables AR reconstruction and guidance directly from monocular arthroscopy without any additional data and/or hardware, our solution may hold the potential for enhancing intraoperative awareness and facilitating surgical precision in arthroscopy. Our AR measurement tool achieves accuracy within 1.59 +/- 1.81mm and the AR annotation tool achieves a mIoU of 0.721.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00299v1">GSPR: Multimodal Place Recognition Using 3D Gaussian Splatting for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-10-01
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Place recognition is a crucial module to ensure autonomous vehicles obtain usable localization information in GPS-denied environments. In recent years, multimodal place recognition methods have gained increasing attention due to their ability to overcome the weaknesses of unimodal sensor systems by leveraging complementary information from different modalities. However, challenges arise from the necessity of harmonizing data across modalities and exploiting the spatio-temporal correlations between them sufficiently. In this paper, we propose a 3D Gaussian Splatting-based multimodal place recognition neural network dubbed GSPR. It explicitly combines multi-view RGB images and LiDAR point clouds into a spatio-temporally unified scene representation with the proposed Multimodal Gaussian Splatting. A network composed of 3D graph convolution and transformer is designed to extract high-level spatio-temporal features and global descriptors from the Gaussian scenes for place recognition. We evaluate our method on the nuScenes dataset, and the experimental results demonstrate that our method can effectively leverage complementary strengths of both multi-view cameras and LiDAR, achieving SOTA place recognition performance while maintaining solid generalization ability. Our open-source code is available at https://github.com/QiZS-BIT/GSPR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.20111v1">Robust Gaussian Splatting SLAM by Leveraging Loop Closure</a></div>
    <div class="paper-meta">
      📅 2024-10-01
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting algorithms excel in novel view rendering applications and have been adapted to extend the capabilities of traditional SLAM systems. However, current Gaussian Splatting SLAM methods, designed mainly for hand-held RGB or RGB-D sensors, struggle with tracking drifts when used with rotating RGB-D camera setups. In this paper, we propose a robust Gaussian Splatting SLAM architecture that utilizes inputs from rotating multiple RGB-D cameras to achieve accurate localization and photorealistic rendering performance. The carefully designed Gaussian Splatting Loop Closure module effectively addresses the issue of accumulated tracking and mapping errors found in conventional Gaussian Splatting SLAM systems. First, each Gaussian is associated with an anchor frame and categorized as historical or novel based on its timestamp. By rendering different types of Gaussians at the same viewpoint, the proposed loop detection strategy considers both co-visibility relationships and distinct rendering outcomes. Furthermore, a loop closure optimization approach is proposed to remove camera pose drift and maintain the high quality of 3D Gaussian models. The approach uses a lightweight pose graph optimization algorithm to correct pose drift and updates Gaussians based on the optimized poses. Additionally, a bundle adjustment scheme further refines camera poses using photometric and geometric constraints, ultimately enhancing the global consistency of scenarios. Quantitative and qualitative evaluations on both synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art methods in camera pose estimation and novel view rendering tasks. The code will be open-sourced for the community.
    </details>
</div>
