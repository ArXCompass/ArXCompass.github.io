# gaussian splatting - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18242v1">GSVisLoc: Generalizable Visual Localization for Gaussian Splatting Scene Representations</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted to ICCV 2025 Workshops (CALIPOSE). Project page: https://gsvisloc.github.io/
    </div>
    <details class="paper-abstract">
      We introduce GSVisLoc, a visual localization method designed for 3D Gaussian Splatting (3DGS) scene representations. Given a 3DGS model of a scene and a query image, our goal is to estimate the camera's position and orientation. We accomplish this by robustly matching scene features to image features. Scene features are produced by downsampling and encoding the 3D Gaussians while image features are obtained by encoding image patches. Our algorithm proceeds in three steps, starting with coarse matching, then fine matching, and finally by applying pose refinement for an accurate final estimate. Importantly, our method leverages the explicit 3DGS scene representation for visual localization without requiring modifications, retraining, or additional reference images. We evaluate GSVisLoc on both indoor and outdoor scenes, demonstrating competitive localization performance on standard benchmarks while outperforming existing 3DGS-based baselines. Moreover, our approach generalizes effectively to novel scenes without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17876v1">Camera Pose Refinement via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Camera pose refinement aims at improving the accuracy of initial pose estimation for applications in 3D computer vision. Most refinement approaches rely on 2D-3D correspondences with specific descriptors or dedicated networks, requiring reconstructing the scene again for a different descriptor or fully retraining the network for each scene. Some recent methods instead infer pose from feature similarity, but their lack of geometry constraints results in less accuracy. To overcome these limitations, we propose a novel camera pose refinement framework leveraging 3D Gaussian Splatting (3DGS), referred to as GS-SMC. Given the widespread usage of 3DGS, our method can employ an existing 3DGS model to render novel views, providing a lightweight solution that can be directly applied to diverse scenes without additional training or fine-tuning. Specifically, we introduce an iterative optimization approach, which refines the camera pose using epipolar geometric constraints among the query and multiple rendered images. Our method allows flexibly choosing feature extractors and matchers to establish these constraints. Extensive empirical evaluations on the 7-Scenes and the Cambridge Landmarks datasets demonstrate that our method outperforms state-of-the-art camera pose refinement approaches, achieving 53.3% and 56.9% reductions in median translation and rotation errors on 7-Scenes, and 40.7% and 53.2% on Cambridge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17811v1">MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 17 pages, 15 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Surface reconstruction has been widely studied in computer vision and graphics. However, existing surface reconstruction works struggle to recover accurate scene geometry when the input views are extremely sparse. To address this issue, we propose MeshSplat, a generalizable sparse-view surface reconstruction framework via Gaussian Splatting. Our key idea is to leverage 2DGS as a bridge, which connects novel view synthesis to learned geometric priors and then transfers these priors to achieve surface reconstruction. Specifically, we incorporate a feed-forward network to predict per-view pixel-aligned 2DGS, which enables the network to synthesize novel view images and thus eliminates the need for direct 3D ground-truth supervision. To improve the accuracy of 2DGS position and orientation prediction, we propose a Weighted Chamfer Distance Loss to regularize the depth maps, especially in overlapping areas of input views, and also a normal prediction network to align the orientation of 2DGS with normal vectors predicted by a monocular normal estimator. Extensive experiments validate the effectiveness of our proposed improvement, demonstrating that our method achieves state-of-the-art performance in generalizable sparse-view mesh reconstruction tasks. Project Page: https://hanzhichang.github.io/meshsplat_web
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15169v2">MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Mesh models have become increasingly accessible for numerous cities; however, the lack of realistic textures restricts their application in virtual urban navigation and autonomous driving. To address this, this paper proposes MeSS (Meshbased Scene Synthesis) for generating high-quality, styleconsistent outdoor scenes with city mesh models serving as the geometric prior. While image and video diffusion models can leverage spatial layouts (such as depth maps or HD maps) as control conditions to generate street-level perspective views, they are not directly applicable to 3D scene generation. Video diffusion models excel at synthesizing consistent view sequences that depict scenes but often struggle to adhere to predefined camera paths or align accurately with rendered control videos. In contrast, image diffusion models, though unable to guarantee cross-view visual consistency, can produce more geometry-aligned results when combined with ControlNet. Building on this insight, our approach enhances image diffusion models by improving cross-view consistency. The pipeline comprises three key stages: first, we generate geometrically consistent sparse views using Cascaded Outpainting ControlNets; second, we propagate denser intermediate views via a component dubbed AGInpaint; and third, we globally eliminate visual inconsistencies (e.g., varying exposure) using the GCAlign module. Concurrently with generation, a 3D Gaussian Splatting (3DGS) scene is reconstructed by initializing Gaussian balls on the mesh surface. Our method outperforms existing approaches in both geometric alignment and generation quality. Once synthesized, the scene can be rendered in diverse styles through relighting and style transfer techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17288v4">GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Occupancy estimation has become a prominent task in 3D computer vision, particularly within the autonomous driving community. In this paper, we present a novel approach to occupancy estimation, termed GaussianFlowOcc, which is inspired by Gaussian Splatting and replaces traditional dense voxel grids with a sparse 3D Gaussian representation. Our efficient model architecture based on a Gaussian Transformer significantly reduces computational and memory requirements by eliminating the need for expensive 3D convolutions used with inefficient voxel-based representations that predominantly represent empty 3D spaces. GaussianFlowOcc effectively captures scene dynamics by estimating temporal flow for each Gaussian during the overall network training process, offering a straightforward solution to a complex problem that is often neglected by existing methods. Moreover, GaussianFlowOcc is designed for scalability, as it employs weak supervision and does not require costly dense 3D voxel annotations based on additional data (e.g., LiDAR). Through extensive experimentation, we demonstrate that GaussianFlowOcc significantly outperforms all previous methods for weakly supervised occupancy estimation on the nuScenes dataset while featuring an inference speed that is 50 times faster than current SOTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17645v1">Generating Human-AI Collaborative Design Sequence for 3D Assets via Differentiable Operation Graph</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      The emergence of 3D artificial intelligence-generated content (3D-AIGC) has enabled rapid synthesis of intricate geometries. However, a fundamental disconnect persists between AI-generated content and human-centric design paradigms, rooted in representational incompatibilities: conventional AI frameworks predominantly manipulate meshes or neural representations (\emph{e.g.}, NeRF, Gaussian Splatting), while designers operate within parametric modeling tools. This disconnection diminishes the practical value of AI for 3D industry, undermining the efficiency of human-AI collaboration. To resolve this disparity, we focus on generating design operation sequences, which are structured modeling histories that comprehensively capture the step-by-step construction process of 3D assets and align with designers' typical workflows in modern 3D software. We first reformulate fundamental modeling operations (\emph{e.g.}, \emph{Extrude}, \emph{Boolean}) into differentiable units, enabling joint optimization of continuous (\emph{e.g.}, \emph{Extrude} height) and discrete (\emph{e.g.}, \emph{Boolean} type) parameters via gradient-based learning. Based on these differentiable operations, a hierarchical graph with gating mechanism is constructed and optimized end-to-end by minimizing Chamfer Distance to target geometries. Multi-stage sequence length constraint and domain rule penalties enable unsupervised learning of compact design sequences without ground-truth sequence supervision. Extensive validation demonstrates that the generated operation sequences achieve high geometric fidelity, smooth mesh wiring, rational step composition and flexible editing capacity, with full compatibility within design industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15376v2">DriveSplat: Decoupled Driving Scene Reconstruction with Geometry-enhanced Partitioned Neural Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      In the realm of driving scenarios, the presence of rapidly moving vehicles, pedestrians in motion, and large-scale static backgrounds poses significant challenges for 3D scene reconstruction. Recent methods based on 3D Gaussian Splatting address the motion blur problem by decoupling dynamic and static components within the scene. However, these decoupling strategies overlook background optimization with adequate geometry relationships and rely solely on fitting each training view by adding Gaussians. Therefore, these models exhibit limited robustness in rendering novel views and lack an accurate geometric representation. To address the above issues, we introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic actors, whose parameters are temporally adjusted by a learnable deformation network. The entire framework is further supervised by depth and normal priors from pre-trained models, improving the accuracy of geometric structures. Our method has been rigorously evaluated on the Waymo and KITTI datasets, demonstrating state-of-the-art performance in novel-view synthesis for driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19938v3">Mesh-Learner: Texturing Mesh with Spherical Harmonics</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 IROS2025 Accepted
    </div>
    <details class="paper-abstract">
      In this paper, we present a 3D reconstruction and rendering framework termed Mesh-Learner that is natively compatible with traditional rasterization pipelines. It integrates mesh and spherical harmonic (SH) texture (i.e., texture filled with SH coefficients) into the learning process to learn each mesh s view-dependent radiance end-to-end. Images are rendered by interpolating surrounding SH Texels at each pixel s sampling point using a novel interpolation method. Conversely, gradients from each pixel are back-propagated to the related SH Texels in SH textures. Mesh-Learner exploits graphic features of rasterization pipeline (texture sampling, deferred rendering) to render, which makes Mesh-Learner naturally compatible with tools (e.g., Blender) and tasks (e.g., 3D reconstruction, scene rendering, reinforcement learning for robotics) that are based on rasterization pipelines. Our system can train vast, unlimited scenes because we transfer only the SH textures within the frustum to the GPU for training. At other times, the SH textures are stored in CPU RAM, which results in moderate GPU memory usage. The rendering results on interpolation and extrapolation sequences in the Replica and FAST-LIVO2 datasets achieve state-of-the-art performance compared to existing state-of-the-art methods (e.g., 3D Gaussian Splatting and M2-Mapping). To benefit the society, the code will be available at https://github.com/hku-mars/Mesh-Learner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17600v1">GWM: Towards Scalable Gaussian World Models for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Published at ICCV 2025. Project page: https://gaussian-world-model.github.io/
    </div>
    <details class="paper-abstract">
      Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17579v1">IDU: Incremental Dynamic Update of Existing 3D Virtual Environments with New Imagery Data</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      For simulation and training purposes, military organizations have made substantial investments in developing high-resolution 3D virtual environments through extensive imaging and 3D scanning. However, the dynamic nature of battlefield conditions-where objects may appear or vanish over time-makes frequent full-scale updates both time-consuming and costly. In response, we introduce the Incremental Dynamic Update (IDU) pipeline, which efficiently updates existing 3D reconstructions, such as 3D Gaussian Splatting (3DGS), with only a small set of newly acquired images. Our approach starts with camera pose estimation to align new images with the existing 3D model, followed by change detection to pinpoint modifications in the scene. A 3D generative AI model is then used to create high-quality 3D assets of the new elements, which are seamlessly integrated into the existing 3D model. The IDU pipeline incorporates human guidance to ensure high accuracy in object identification and placement, with each update focusing on a single new object at a time. Experimental results confirm that our proposed IDU pipeline significantly reduces update time and labor, offering a cost-effective and targeted solution for maintaining up-to-date 3D models in rapidly evolving military scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17012v1">Fiducial Marker Splatting for High-Fidelity Robotics Simulations</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      High-fidelity 3D simulation is critical for training mobile robots, but its traditional reliance on mesh-based representations often struggle in complex environments, such as densely packed greenhouses featuring occlusions and repetitive structures. Recent neural rendering methods, like Gaussian Splatting (GS), achieve remarkable visual realism but lack flexibility to incorporate fiducial markers, which are essential for robotic localization and control. We propose a hybrid framework that combines the photorealism of GS with structured marker representations. Our core contribution is a novel algorithm for efficiently generating GS-based fiducial markers (e.g., AprilTags) within cluttered scenes. Experiments show that our approach outperforms traditional image-fitting techniques in both efficiency and pose-estimation accuracy. We further demonstrate the framework's potential in a greenhouse simulation. This agricultural setting serves as a challenging testbed, as its combination of dense foliage, similar-looking elements, and occlusions pushes the limits of perception, thereby highlighting the framework's value for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18561v3">Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and Real-time Rendering</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Project page: https://fudan-zvg.github.io/PVG/
    </div>
    <details class="paper-abstract">
      Modeling dynamic, large-scale urban scenes is challenging due to their highly intricate geometric structures and unconstrained dynamics in both space and time. Prior methods often employ high-level architectural priors, separating static and dynamic elements, resulting in suboptimal capture of their synergistic interactions. To address this challenge, we present a unified representation model, called Periodic Vibration Gaussian (PVG). PVG builds upon the efficient 3D Gaussian splatting technique, originally designed for static scene representation, by introducing periodic vibration-based temporal dynamics. This innovation enables PVG to elegantly and uniformly represent the characteristics of various objects and elements in dynamic urban scenes. To enhance temporally coherent and large scene representation learning with sparse training data, we introduce a novel temporal smoothing mechanism and a position-aware adaptive control strategy respectively. Extensive experiments on Waymo Open Dataset and KITTI benchmarks demonstrate that PVG surpasses state-of-the-art alternatives in both reconstruction and novel view synthesis for both dynamic and static scenes. Notably, PVG achieves this without relying on manually labeled object bounding boxes or expensive optical flow estimation. Moreover, PVG exhibits 900-fold acceleration in rendering over the best alternative.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16849v1">RF-PGS: Fully-structured Spatial Wireless Channel Representation with Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 13 pages, 16 figures, in submission to IEEE journal
    </div>
    <details class="paper-abstract">
      In the 6G era, the demand for higher system throughput and the implementation of emerging 6G technologies require large-scale antenna arrays and accurate spatial channel state information (Spatial-CSI). Traditional channel modeling approaches, such as empirical models, ray tracing, and measurement-based methods, face challenges in spatial resolution, efficiency, and scalability. Radiance field-based methods have emerged as promising alternatives but still suffer from geometric inaccuracy and costly supervision. This paper proposes RF-PGS, a novel framework that reconstructs high-fidelity radio propagation paths from only sparse path loss spectra. By introducing Planar Gaussians as geometry primitives with certain RF-specific optimizations, RF-PGS achieves dense, surface-aligned scene reconstruction in the first geometry training stage. In the subsequent Radio Frequency (RF) training stage, the proposed fully-structured radio radiance, combined with a tailored multi-view loss, accurately models radio propagation behavior. Compared to prior radiance field methods, RF-PGS significantly improves reconstruction accuracy, reduces training costs, and enables efficient representation of wireless channels, offering a practical solution for scalable 6G Spatial-CSI modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16467v1">Arbitrary-Scale 3D Gaussian Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting (3DGS) super-resolution methods typically perform high-resolution (HR) rendering of fixed scale factors, making them impractical for resource-limited scenarios. Directly rendering arbitrary-scale HR views with vanilla 3DGS introduces aliasing artifacts due to the lack of scale-aware rendering ability, while adding a post-processing upsampler for 3DGS complicates the framework and reduces rendering efficiency. To tackle these issues, we build an integrated framework that incorporates scale-aware rendering, generative prior-guided optimization, and progressive super-resolving to enable 3D Gaussian super-resolution of arbitrary scale factors with a single 3D model. Notably, our approach supports both integer and non-integer scale rendering to provide more flexibility. Extensive experiments demonstrate the effectiveness of our model in rendering high-quality arbitrary-scale HR views (6.59 dB PSNR gain over 3DGS) with a single model. It preserves structural consistency with LR views and across different scales, while maintaining real-time rendering speed (85 FPS at 1080p).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09977v2">A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 GitHub Repo: https://github.com/heshuting555/Awesome-3DGS-Applications
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful alternative to Neural Radiance Fields (NeRF) for 3D scene representation, offering high-fidelity photorealistic rendering with real-time performance. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first introduces 2D foundation models that support semantic understanding and control in 3DGS applications, followed by a review of NeRF-based methods that inform their 3DGS counterparts. We then categorize 3DGS applications into segmentation, editing, generation, and other functional tasks. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at https://github.com/heshuting555/Awesome-3DGS-Applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03890v7">A Survey on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Ongoing project; Paper list: https://github.com/guikunchen/Awesome3DGS ; Benchmark: https://github.com/guikunchen/3DGS-Benchmarks
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (GS) has emerged as a transformative technique in explicit radiance field and computer graphics. This innovative approach, characterized by the use of millions of learnable 3D Gaussians, represents a significant departure from mainstream neural radiance field approaches, which predominantly use implicit, coordinate-based models to map spatial coordinates to pixel values. 3D GS, with its explicit scene representation and differentiable rendering algorithm, not only promises real-time rendering capability but also introduces unprecedented levels of editability. This positions 3D GS as a potential game-changer for the next generation of 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in the domain of 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in explicit radiance field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02146v2">ScrewSplat: An End-to-End Method for Articulated Object Recognition</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 26 pages, 12 figures, Conference on Robot Learning (CoRL) 2025
    </div>
    <details class="paper-abstract">
      Articulated object recognition -- the task of identifying both the geometry and kinematic joints of objects with movable parts -- is essential for enabling robots to interact with everyday objects such as doors and laptops. However, existing approaches often rely on strong assumptions, such as a known number of articulated parts; require additional inputs, such as depth images; or involve complex intermediate steps that can introduce potential errors -- limiting their practicality in real-world settings. In this paper, we introduce ScrewSplat, a simple end-to-end method that operates solely on RGB observations. Our approach begins by randomly initializing screw axes, which are then iteratively optimized to recover the object's underlying kinematic structure. By integrating with Gaussian Splatting, we simultaneously reconstruct the 3D geometry and segment the object into rigid, movable parts. We demonstrate that our method achieves state-of-the-art recognition accuracy across a diverse set of articulated objects, and further enables zero-shot, text-guided manipulation using the recovered kinematic model. See the project website at: https://screwsplat.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00083v2">Localized Gaussian Splatting Editing with Contextual Awareness</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 WACV 2025
    </div>
    <details class="paper-abstract">
      Recent text-guided generation of individual 3D object has achieved great success using diffusion priors. However, these methods are not suitable for object insertion and replacement tasks as they do not consider the background, leading to illumination mismatches within the environment. To bridge the gap, we introduce an illumination-aware 3D scene editing pipeline for 3D Gaussian Splatting (3DGS) representation. Our key observation is that inpainting by the state-of-the-art conditional 2D diffusion model is consistent with background in lighting. To leverage the prior knowledge from the well-trained diffusion models for 3D object generation, our approach employs a coarse-to-fine objection optimization pipeline with inpainted views. In the first coarse step, we achieve image-to-3D lifting given an ideal inpainted view. The process employs 3D-aware diffusion prior from a view-conditioned diffusion model, which preserves illumination present in the conditioning image. To acquire an ideal inpainted image, we introduce an Anchor View Proposal (AVP) algorithm to find a single view that best represents the scene illumination in target region. In the second Texture Enhancement step, we introduce a novel Depth-guided Inpainting Score Distillation Sampling (DI-SDS), which enhances geometry and texture details with the inpainting diffusion prior, beyond the scope of the 3D-aware diffusion prior knowledge in the first coarse step. DI-SDS not only provides fine-grained texture enhancement, but also urges optimization to respect scene lighting. Our approach efficiently achieves local editing with global illumination consistency without explicitly modeling light transport. We demonstrate robustness of our method by evaluating editing in real scenes containing explicit highlight and shadows, and compare against the state-of-the-art text-to-3D editing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10133v4">Efficient Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Resubmission version in arxiv at arXiv:2508.12313
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated outstanding performance in novel view synthesis, achieving a balance between rendering quality and real-time performance. 3DGS employs Adaptive Density Control (ADC) to increase the number of Gaussians. However, the clone and split operations within ADC are not sufficiently efficient, impacting optimization speed and detail recovery. Additionally, overfitted Gaussians that affect rendering quality may exist, and the original ADC is unable to remove them. To address these issues, we propose two key innovations: (1) Long-Axis Split, which precisely controls the position, shape, and opacity of child Gaussians to minimize the difference before and after splitting. (2) Recovery-Aware Pruning, which leverages differences in recovery speed after resetting opacity to prune overfitted Gaussians, thereby improving generalization performance. Experimental results show that our method significantly enhances rendering quality. Due to resubmission reasons, this version has been abandoned. The improved version is available at https://xiaobin2001.github.io/improved-gs-web .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12415v2">TiP4GEN: Text to Immersive Panorama 4D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted In Proceedings of the 33rd ACM International Conference on Multimedia (MM' 25)
    </div>
    <details class="paper-abstract">
      With the rapid advancement and widespread adoption of VR/AR technologies, there is a growing demand for the creation of high-quality, immersive dynamic scenes. However, existing generation works predominantly concentrate on the creation of static scenes or narrow perspective-view dynamic scenes, falling short of delivering a truly 360-degree immersive experience from any viewpoint. In this paper, we introduce \textbf{TiP4GEN}, an advanced text-to-dynamic panorama scene generation framework that enables fine-grained content control and synthesizes motion-rich, geometry-consistent panoramic 4D scenes. TiP4GEN integrates panorama video generation and dynamic scene reconstruction to create 360-degree immersive virtual environments. For video generation, we introduce a \textbf{Dual-branch Generation Model} consisting of a panorama branch and a perspective branch, responsible for global and local view generation, respectively. A bidirectional cross-attention mechanism facilitates comprehensive information exchange between the branches. For scene reconstruction, we propose a \textbf{Geometry-aligned Reconstruction Model} based on 3D Gaussian Splatting. By aligning spatial-temporal point clouds using metric depth maps and initializing scene cameras with estimated poses, our method ensures geometric consistency and temporal coherence for the reconstructed scenes. Extensive experiments demonstrate the effectiveness of our proposed designs and the superiority of TiP4GEN in generating visually compelling and motion-coherent dynamic panoramic scenes. Our project page is at https://ke-xing.github.io/TiP4GEN/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12892v2">3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted to ICCV 2025. Project page: https://lukashoel.github.io/3DGS-LM, Video: https://www.youtube.com/watch?v=tDiGuGMssg8, Code: https://github.com/lukasHoel/3DGS-LM
    </div>
    <details class="paper-abstract">
      We present 3DGS-LM, a new method that accelerates the reconstruction of 3D Gaussian Splatting (3DGS) by replacing its ADAM optimizer with a tailored Levenberg-Marquardt (LM). Existing methods reduce the optimization time by decreasing the number of Gaussians or by improving the implementation of the differentiable rasterizer. However, they still rely on the ADAM optimizer to fit Gaussian parameters of a scene in thousands of iterations, which can take up to an hour. To this end, we change the optimizer to LM that runs in conjunction with the 3DGS differentiable rasterizer. For efficient GPU parallization, we propose a caching data structure for intermediate gradients that allows us to efficiently calculate Jacobian-vector products in custom CUDA kernels. In every LM iteration, we calculate update directions from multiple image subsets using these kernels and combine them in a weighted mean. Overall, our method is 20% faster than the original 3DGS while obtaining the same reconstruction quality. Our optimization is also agnostic to other methods that acclerate 3DGS, thus enabling even faster speedups compared to vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15457v1">Enhancing Novel View Synthesis from extremely sparse views with SfM-free 3D Gaussian Splatting Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable real-time performance in novel view synthesis, yet its effectiveness relies heavily on dense multi-view inputs with precisely known camera poses, which are rarely available in real-world scenarios. When input views become extremely sparse, the Structure-from-Motion (SfM) method that 3DGS depends on for initialization fails to accurately reconstruct the 3D geometric structures of scenes, resulting in degraded rendering quality. In this paper, we propose a novel SfM-free 3DGS-based method that jointly estimates camera poses and reconstructs 3D scenes from extremely sparse-view inputs. Specifically, instead of SfM, we propose a dense stereo module to progressively estimates camera pose information and reconstructs a global dense point cloud for initialization. To address the inherent problem of information scarcity in extremely sparse-view settings, we propose a coherent view interpolation module that interpolates camera poses based on training view pairs and generates viewpoint-consistent content as additional supervision signals for training. Furthermore, we introduce multi-scale Laplacian consistent regularization and adaptive spatial-aware multi-scale geometry regularization to enhance the quality of geometrical structures and rendered content. Experiments show that our method significantly outperforms other state-of-the-art 3DGS-based approaches, achieving a remarkable 2.75dB improvement in PSNR under extremely sparse-view conditions (using only 2 training views). The images synthesized by our method exhibit minimal distortion while preserving rich high-frequency details, resulting in superior visual quality compared to existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14278v2">GALA: Guided Attention with Language Alignment for Open Vocabulary Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      3D scene reconstruction and understanding have gained increasing popularity, yet existing methods still struggle to capture fine-grained, language-aware 3D representations from 2D images. In this paper, we present GALA, a novel framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS). GALA distills a scene-specific 3D instance feature field via self-supervised contrastive learning. To extend to generalized language feature fields, we introduce the core contribution of GALA, a cross-attention module with two learnable codebooks that encode view-independent semantic embeddings. This design not only ensures intra-instance feature similarity but also supports seamless 2D and 3D open-vocabulary queries. It reduces memory consumption by avoiding per-Gaussian high-dimensional feature learning. Extensive experiments on real-world datasets demonstrate GALA's remarkable open-vocabulary performance on both 2D and 3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15376v1">DriveSplat: Decoupled Driving Scene Reconstruction with Geometry-enhanced Partitioned Neural Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      In the realm of driving scenarios, the presence of rapidly moving vehicles, pedestrians in motion, and large-scale static backgrounds poses significant challenges for 3D scene reconstruction. Recent methods based on 3D Gaussian Splatting address the motion blur problem by decoupling dynamic and static components within the scene. However, these decoupling strategies overlook background optimization with adequate geometry relationships and rely solely on fitting each training view by adding Gaussians. Therefore, these models exhibit limited robustness in rendering novel views and lack an accurate geometric representation. To address the above issues, we introduce DriveSplat, a high-quality reconstruction method for driving scenarios based on neural Gaussian representations with dynamic-static decoupling. To better accommodate the predominantly linear motion patterns of driving viewpoints, a region-wise voxel initialization scheme is employed, which partitions the scene into near, middle, and far regions to enhance close-range detail representation. Deformable neural Gaussians are introduced to model non-rigid dynamic actors, whose parameters are temporally adjusted by a learnable deformation network. The entire framework is further supervised by depth and normal priors from pre-trained models, improving the accuracy of geometric structures. Our method has been rigorously evaluated on the Waymo and KITTI datasets, demonstrating state-of-the-art performance in novel-view synthesis for driving scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15372v1">Image-Conditioned 3D Gaussian Splat Quantization</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has attracted considerable attention for enabling high-quality real-time rendering. Although 3DGS compression methods have been proposed for deployment on storage-constrained devices, two limitations hinder archival use: (1) they compress medium-scale scenes only to the megabyte range, which remains impractical for large-scale scenes or extensive scene collections; and (2) they lack mechanisms to accommodate scene changes after long-term archival. To address these limitations, we propose an Image-Conditioned Gaussian Splat Quantizer (ICGS-Quantizer) that substantially enhances compression efficiency and provides adaptability to scene changes after archiving. ICGS-Quantizer improves quantization efficiency by jointly exploiting inter-Gaussian and inter-attribute correlations and by using shared codebooks across all training scenes, which are then fixed and applied to previously unseen test scenes, eliminating the overhead of per-scene codebooks. This approach effectively reduces the storage requirements for 3DGS to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions scene decoding on images captured at decoding time. The encoding, quantization, and decoding processes are trained jointly, ensuring that the codes, which are quantized representations of the scene, are effective for conditional decoding. We evaluate ICGS-Quantizer on 3D scene compression and 3D scene updating. Experimental results show that ICGS-Quantizer consistently outperforms state-of-the-art methods in compression efficiency and adaptability to scene changes. Our code, model, and data will be publicly available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09588v2">TextSplat: Text-Guided Semantic Fusion for Generalizable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Recent advancements in Generalizable Gaussian Splatting have enabled robust 3D reconstruction from sparse input views by utilizing feed-forward Gaussian Splatting models, achieving superior cross-scene generalization. However, while many methods focus on geometric consistency, they often neglect the potential of text-driven guidance to enhance semantic understanding, which is crucial for accurately reconstructing fine-grained details in complex scenes. To address this limitation, we propose TextSplat--the first text-driven Generalizable Gaussian Splatting framework. By employing a text-guided fusion of diverse semantic cues, our framework learns robust cross-modal feature representations that improve the alignment of geometric and semantic information, producing high-fidelity 3D reconstructions. Specifically, our framework employs three parallel modules to obtain complementary representations: the Diffusion Prior Depth Estimator for accurate depth information, the Semantic Aware Segmentation Network for detailed semantic information, and the Multi-View Interaction Network for refined cross-view features. Then, in the Text-Guided Semantic Fusion Module, these representations are integrated via the text-guided and attention-based feature aggregation mechanism, resulting in enhanced 3D Gaussian parameters enriched with detailed semantic cues. Experimental results on various benchmark datasets demonstrate improved performance compared to existing methods across multiple evaluation metrics, validating the effectiveness of our framework. The code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15169v1">MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Mesh models have become increasingly accessible for numerous cities; however, the lack of realistic textures restricts their application in virtual urban navigation and autonomous driving. To address this, this paper proposes MeSS (Meshbased Scene Synthesis) for generating high-quality, styleconsistent outdoor scenes with city mesh models serving as the geometric prior. While image and video diffusion models can leverage spatial layouts (such as depth maps or HD maps) as control conditions to generate street-level perspective views, they are not directly applicable to 3D scene generation. Video diffusion models excel at synthesizing consistent view sequences that depict scenes but often struggle to adhere to predefined camera paths or align accurately with rendered control videos. In contrast, image diffusion models, though unable to guarantee cross-view visual consistency, can produce more geometry-aligned results when combined with ControlNet. Building on this insight, our approach enhances image diffusion models by improving cross-view consistency. The pipeline comprises three key stages: first, we generate geometrically consistent sparse views using Cascaded Outpainting ControlNets; second, we propagate denser intermediate views via a component dubbed AGInpaint; and third, we globally eliminate visual inconsistencies (e.g., varying exposure) using the GCAlign module. Concurrently with generation, a 3D Gaussian Splatting (3DGS) scene is reconstructed by initializing Gaussian balls on the mesh surface. Our method outperforms existing approaches in both geometric alignment and generation quality. Once synthesized, the scene can be rendered in diverse styles through relighting and style transfer techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15151v1">Zero-shot Volumetric CT Super-Resolution using 3D Gaussian Splatting with Upsampled 2D X-ray Projection Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Computed tomography (CT) is widely used in clinical diagnosis, but acquiring high-resolution (HR) CT is limited by radiation exposure risks. Deep learning-based super-resolution (SR) methods have been studied to reconstruct HR from low-resolution (LR) inputs. While supervised SR approaches have shown promising results, they require large-scale paired LR-HR volume datasets that are often unavailable. In contrast, zero-shot methods alleviate the need for paired data by using only a single LR input, but typically struggle to recover fine anatomical details due to limited internal information. To overcome these, we propose a novel zero-shot 3D CT SR framework that leverages upsampled 2D X-ray projection priors generated by a diffusion model. Exploiting the abundance of HR 2D X-ray data, we train a diffusion model on large-scale 2D X-ray projection and introduce a per-projection adaptive sampling strategy. It selects the generative process for each projection, thus providing HR projections as strong external priors for 3D CT reconstruction. These projections serve as inputs to 3D Gaussian splatting for reconstructing a 3D CT volume. Furthermore, we propose negative alpha blending (NAB-GS) that allows negative values in Gaussian density representation. NAB-GS enables residual learning between LR and diffusion-based projections, thereby enhancing high-frequency structure reconstruction. Experiments on two datasets show that our method achieves superior quantitative and qualitative results for 3D CT SR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15972v1">UnPose: Uncertainty-Guided Diffusion Priors for Zero-Shot Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Published at the Conference on Robot Learning (CoRL) 2025. For more details please visit https://frankzhaodong.github.io/UnPose
    </div>
    <details class="paper-abstract">
      Estimating the 6D pose of novel objects is a fundamental yet challenging problem in robotics, often relying on access to object CAD models. However, acquiring such models can be costly and impractical. Recent approaches aim to bypass this requirement by leveraging strong priors from foundation models to reconstruct objects from single or multi-view images, but typically require additional training or produce hallucinated geometry. To this end, we propose UnPose, a novel framework for zero-shot, model-free 6D object pose estimation and reconstruction that exploits 3D priors and uncertainty estimates from a pre-trained diffusion model. Specifically, starting from a single-view RGB-D frame, UnPose uses a multi-view diffusion model to estimate an initial 3D model using 3D Gaussian Splatting (3DGS) representation, along with pixel-wise epistemic uncertainty estimates. As additional observations become available, we incrementally refine the 3DGS model by fusing new views guided by the diffusion model's uncertainty, thereby continuously improving the pose estimation accuracy and 3D reconstruction quality. To ensure global consistency, the diffusion prior-generated views and subsequent observations are further integrated in a pose graph and jointly optimized into a coherent 3DGS field. Extensive experiments demonstrate that UnPose significantly outperforms existing approaches in both 6D pose estimation accuracy and 3D reconstruction quality. We further showcase its practical applicability in real-world robotic manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14717v1">GSFix3D: Diffusion-Guided Repair of Novel Views in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Recent developments in 3D Gaussian Splatting have significantly enhanced novel view synthesis, yet generating high-quality renderings from extreme novel viewpoints or partially observed regions remains challenging. Meanwhile, diffusion models exhibit strong generative capabilities, but their reliance on text prompts and lack of awareness of specific scene information hinder accurate 3D reconstruction tasks. To address these limitations, we introduce GSFix3D, a novel framework that improves the visual fidelity in under-constrained regions by distilling prior knowledge from diffusion models into 3D representations, while preserving consistency with observed scene details. At its core is GSFixer, a latent diffusion model obtained via our customized fine-tuning protocol that can leverage both mesh and 3D Gaussians to adapt pretrained generative models to a variety of environments and artifact types from different reconstruction methods, enabling robust novel view repair for unseen camera poses. Moreover, we propose a random mask augmentation strategy that empowers GSFixer to plausibly inpaint missing regions. Experiments on challenging benchmarks demonstrate that our GSFix3D and GSFixer achieve state-of-the-art performance, requiring only minimal scene-specific fine-tuning on captured data. Real-world test further confirms its resilience to potential pose errors. Our code and data will be made publicly available. Project page: https://gsfix3d.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06926v3">Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 ICRA 2025
    </div>
    <details class="paper-abstract">
      In this paper, we present a real-time photo-realistic SLAM method based on marrying Gaussian Splatting with LiDAR-Inertial-Camera SLAM. Most existing radiance-field-based SLAM systems mainly focus on bounded indoor environments, equipped with RGB-D or RGB sensors. However, they are prone to decline when expanding to unbounded scenes or encountering adverse conditions, such as violent motions and changing illumination. In contrast, oriented to general scenarios, our approach additionally tightly fuses LiDAR, IMU, and camera for robust pose estimation and photo-realistic online mapping. To compensate for regions unobserved by the LiDAR, we propose to integrate both the triangulated visual points from images and LiDAR points for initializing 3D Gaussians. In addition, the modeling of the sky and varying camera exposure have been realized for high-quality rendering. Notably, we implement our system purely with C++ and CUDA, and meticulously design a series of strategies to accelerate the online optimization of the Gaussian-based scene representation. Extensive experiments demonstrate that our method outperforms its counterparts while maintaining real-time capability. Impressively, regarding photo-realistic mapping, our method with our estimated poses even surpasses all the compared approaches that utilize privileged ground-truth poses for mapping. Our code has been released on https://github.com/APRIL-ZJU/Gaussian-LIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14682v1">GeMS: Efficient Gaussian Splatting for Extreme Motion Blur</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      We introduce GeMS, a framework for 3D Gaussian Splatting (3DGS) designed to handle severely motion-blurred images. State-of-the-art deblurring methods for extreme blur, such as ExBluRF, as well as Gaussian Splatting-based approaches like Deblur-GS, typically assume access to sharp images for camera pose estimation and point cloud generation, an unrealistic assumption. Methods relying on COLMAP initialization, such as BAD-Gaussians, also fail due to unreliable feature correspondences under severe blur. To address these challenges, we propose GeMS, a 3DGS framework that reconstructs scenes directly from extremely blurred images. GeMS integrates: (1) VGGSfM, a deep learning-based Structure-from-Motion pipeline that estimates poses and generates point clouds directly from blurred inputs; (2) 3DGS-MCMC, which enables robust scene initialization by treating Gaussians as samples from a probability distribution, eliminating heuristic densification and pruning; and (3) joint optimization of camera trajectories and Gaussian parameters for stable reconstruction. While this pipeline produces strong results, inaccuracies may remain when all inputs are severely blurred. To mitigate this, we propose GeMS-E, which integrates a progressive refinement step using events: (4) Event-based Double Integral (EDI) deblurring restores sharper images that are then fed into GeMS, improving pose estimation, point cloud generation, and overall reconstruction. Both GeMS and GeMS-E achieve state-of-the-art performance on synthetic and real-world datasets. To our knowledge, this is the first framework to address extreme motion blur within 3DGS directly from severely blurred inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14563v1">GOGS: High-Fidelity Geometry and Relighting for Glossy Objects via Gaussian Surfels</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 13 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Inverse rendering of glossy objects from RGB imagery remains fundamentally limited by inherent ambiguity. Although NeRF-based methods achieve high-fidelity reconstruction via dense-ray sampling, their computational cost is prohibitive. Recent 3D Gaussian Splatting achieves high reconstruction efficiency but exhibits limitations under specular reflections. Multi-view inconsistencies introduce high-frequency surface noise and structural artifacts, while simplified rendering equations obscure material properties, leading to implausible relighting results. To address these issues, we propose GOGS, a novel two-stage framework based on 2D Gaussian surfels. First, we establish robust surface reconstruction through physics-based rendering with split-sum approximation, enhanced by geometric priors from foundation models. Second, we perform material decomposition by leveraging Monte Carlo importance sampling of the full rendering equation, modeling indirect illumination via differentiable 2D Gaussian ray tracing and refining high-frequency specular details through spherical mipmap-based directional encoding that captures anisotropic highlights. Extensive experiments demonstrate state-of-the-art performance in geometry reconstruction, material separation, and photorealistic relighting under novel illuminations, outperforming existing inverse rendering approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14552v1">From Slices to Structures: Unsupervised 3D Reconstruction of Female Pelvic Anatomy from Freehand Transvaginal Ultrasound</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Volumetric ultrasound has the potential to significantly improve diagnostic accuracy and clinical decision-making, yet its widespread adoption remains limited by dependence on specialized hardware and restrictive acquisition protocols. In this work, we present a novel unsupervised framework for reconstructing 3D anatomical structures from freehand 2D transvaginal ultrasound (TVS) sweeps, without requiring external tracking or learned pose estimators. Our method adapts the principles of Gaussian Splatting to the domain of ultrasound, introducing a slice-aware, differentiable rasterizer tailored to the unique physics and geometry of ultrasound imaging. We model anatomy as a collection of anisotropic 3D Gaussians and optimize their parameters directly from image-level supervision, leveraging sensorless probe motion estimation and domain-specific geometric priors. The result is a compact, flexible, and memory-efficient volumetric representation that captures anatomical detail with high spatial fidelity. This work demonstrates that accurate 3D reconstruction from 2D ultrasound images can be achieved through purely computational means, offering a scalable alternative to conventional 3D systems and enabling new opportunities for AI-assisted analysis and diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14443v1">Reconstruction Using the Invisible: Intuition from NIR and Metadata for Enhanced 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has rapidly advanced, its application in agriculture remains underexplored. Agricultural scenes present unique challenges for 3D reconstruction methods, particularly due to uneven illumination, occlusions, and a limited field of view. To address these limitations, we introduce \textbf{NIRPlant}, a novel multimodal dataset encompassing Near-Infrared (NIR) imagery, RGB imagery, textual metadata, Depth, and LiDAR data collected under varied indoor and outdoor lighting conditions. By integrating NIR data, our approach enhances robustness and provides crucial botanical insights that extend beyond the visible spectrum. Additionally, we leverage text-based metadata derived from vegetation indices, such as NDVI, NDWI, and the chlorophyll index, which significantly enriches the contextual understanding of complex agricultural environments. To fully exploit these modalities, we propose \textbf{NIRSplat}, an effective multimodal Gaussian splatting architecture employing a cross-attention mechanism combined with 3D point-based positional encoding, providing robust geometric priors. Comprehensive experiments demonstrate that \textbf{NIRSplat} outperforms existing landmark methods, including 3DGS, CoR-GS, and InstantSplat, highlighting its effectiveness in challenging agricultural scenarios. The code and dataset are publicly available at: https://github.com/StructuresComp/3D-Reconstruction-NIR
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17437v1">Pixie: Fast and Generalizable Supervised Learning of 3D Physics from Pixels</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Website: https://pixie-3d.github.io/
    </div>
    <details class="paper-abstract">
      Inferring the physical properties of 3D scenes from visual information is a critical yet challenging task for creating interactive and realistic virtual worlds. While humans intuitively grasp material characteristics such as elasticity or stiffness, existing methods often rely on slow, per-scene optimization, limiting their generalizability and application. To address this problem, we introduce PIXIE, a novel method that trains a generalizable neural network to predict physical properties across multiple scenes from 3D visual features purely using supervised losses. Once trained, our feed-forward network can perform fast inference of plausible material fields, which coupled with a learned static scene representation like Gaussian Splatting enables realistic physics simulation under external forces. To facilitate this research, we also collected PIXIEVERSE, one of the largest known datasets of paired 3D assets and physic material annotations. Extensive evaluations demonstrate that PIXIE is about 1.46-4.39x better and orders of magnitude faster than test-time optimization methods. By leveraging pretrained visual features like CLIP, our method can also zero-shot generalize to real-world scenes despite only ever been trained on synthetic data. https://pixie-3d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14041v1">LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 ICCV 2025. Project page: https://linjohnss.github.io/longsplat/
    </div>
    <details class="paper-abstract">
      LongSplat addresses critical challenges in novel view synthesis (NVS) from casually captured long videos characterized by irregular camera motion, unknown camera poses, and expansive scenes. Current methods often suffer from pose drift, inaccurate geometry initialization, and severe memory limitations. To address these issues, we introduce LongSplat, a robust unposed 3D Gaussian Splatting framework featuring: (1) Incremental Joint Optimization that concurrently optimizes camera poses and 3D Gaussians to avoid local minima and ensure global consistency; (2) a robust Pose Estimation Module leveraging learned 3D priors; and (3) an efficient Octree Anchor Formation mechanism that converts dense point clouds into anchors based on spatial density. Extensive experiments on challenging benchmarks demonstrate that LongSplat achieves state-of-the-art results, substantially improving rendering quality, pose accuracy, and computational efficiency compared to prior approaches. Project page: https://linjohnss.github.io/longsplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14037v1">Distilled-3DGS:Distilled 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Project page: https://distilled3dgs.github.io Code: https://github.com/lt-xiang/Distilled-3DGS
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has exhibited remarkable efficacy in novel view synthesis (NVS). However, it suffers from a significant drawback: achieving high-fidelity rendering typically necessitates a large number of 3D Gaussians, resulting in substantial memory consumption and storage requirements. To address this challenge, we propose the first knowledge distillation framework for 3DGS, featuring various teacher models, including vanilla 3DGS, noise-augmented variants, and dropout-regularized versions. The outputs of these teachers are aggregated to guide the optimization of a lightweight student model. To distill the hidden geometric structure, we propose a structural similarity loss to boost the consistency of spatial geometric distributions between the student and teacher model. Through comprehensive quantitative and qualitative evaluations across diverse datasets, the proposed Distilled-3DGS, a simple yet effective framework without bells and whistles, achieves promising rendering results in both rendering quality and storage efficiency compared to state-of-the-art methods. Project page: https://distilled3dgs.github.io . Code: https://github.com/lt-xiang/Distilled-3DGS .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14014v1">Online 3D Gaussian Splatting Modeling with Novel View Selection</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      This study addresses the challenge of generating online 3D Gaussian Splatting (3DGS) models from RGB-only frames. Previous studies have employed dense SLAM techniques to estimate 3D scenes from keyframes for 3DGS model construction. However, these methods are limited by their reliance solely on keyframes, which are insufficient to capture an entire scene, resulting in incomplete reconstructions. Moreover, building a generalizable model requires incorporating frames from diverse viewpoints to achieve broader scene coverage. However, online processing restricts the use of many frames or extensive training iterations. Therefore, we propose a novel method for high-quality 3DGS modeling that improves model completeness through adaptive view selection. By analyzing reconstruction quality online, our approach selects optimal non-keyframes for additional training. By integrating both keyframes and selected non-keyframes, the method refines incomplete regions from diverse viewpoints, significantly enhancing completeness. We also present a framework that incorporates an online multi-view stereo approach, ensuring consistency in 3D information throughout the 3DGS modeling process. Experimental results demonstrate that our method outperforms state-of-the-art methods, delivering exceptional performance in complex outdoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10486v2">DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 17 pages, 9 figures, ICCV 2025 Findings Oral, Project pages: https://jzr99.github.io/DNF-Avatar/
    </div>
    <details class="paper-abstract">
      Creating relightable and animatable human avatars from monocular videos is a rising research topic with a range of applications, e.g. virtual reality, sports, and video games. Previous works utilize neural fields together with physically based rendering (PBR), to estimate geometry and disentangle appearance properties of human avatars. However, one drawback of these methods is the slow rendering speed due to the expensive Monte Carlo ray tracing. To tackle this problem, we proposed to distill the knowledge from implicit neural fields (teacher) to explicit 2D Gaussian splatting (student) representation to take advantage of the fast rasterization property of Gaussian splatting. To avoid ray-tracing, we employ the split-sum approximation for PBR appearance. We also propose novel part-wise ambient occlusion probes for shadow computation. Shadow prediction is achieved by querying these probes only once per pixel, which paves the way for real-time relighting of avatars. These techniques combined give high-quality relighting results with realistic shadow effects. Our experiments demonstrate that the proposed student model achieves comparable or even better relighting results with our teacher model while being 370 times faster at inference time, achieving a 67 FPS rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13911v1">PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      While physics-grounded 3D motion synthesis has seen significant progress, current methods face critical limitations. They typically rely on pre-reconstructed 3D Gaussian Splatting (3DGS) representations, while physics integration depends on either inflexible, manually defined physical attributes or unstable, optimization-heavy guidance from video models. To overcome these challenges, we introduce PhysGM, a feed-forward framework that jointly predicts a 3D Gaussian representation and its physical properties from a single image, enabling immediate, physical simulation and high-fidelity 4D rendering. We first establish a base model by jointly optimizing for Gaussian reconstruction and probabilistic physics prediction. The model is then refined with physically plausible reference videos to enhance both rendering fidelity and physics prediction accuracy. We adopt the Direct Preference Optimization (DPO) to align its simulations with reference videos, circumventing Score Distillation Sampling (SDS) optimization which needs back-propagating gradients through the complex differentiable simulation and rasterization. To facilitate the training, we introduce a new dataset PhysAssets of over 24,000 3D assets, annotated with physical properties and corresponding guiding videos. Experimental results demonstrate that our method effectively generates high-fidelity 4D simulations from a single image in one minute. This represents a significant speedup over prior works while delivering realistic rendering results. Our project page is at:https://hihixiaolv.github.io/PhysGM.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13537v1">EAvatar: Expression-Aware Head Avatar Reconstruction with Generative Geometry Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      High-fidelity head avatar reconstruction plays a crucial role in AR/VR, gaming, and multimedia content creation. Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated effectiveness in modeling complex geometry with real-time rendering capability and are now widely used in high-fidelity head avatar reconstruction tasks. However, existing 3DGS-based methods still face significant challenges in capturing fine-grained facial expressions and preserving local texture continuity, especially in highly deformable regions. To mitigate these limitations, we propose a novel 3DGS-based framework termed EAvatar for head reconstruction that is both expression-aware and deformation-aware. Our method introduces a sparse expression control mechanism, where a small number of key Gaussians are used to influence the deformation of their neighboring Gaussians, enabling accurate modeling of local deformations and fine-scale texture transitions. Furthermore, we leverage high-quality 3D priors from pretrained generative models to provide a more reliable facial geometry, offering structural guidance that improves convergence stability and shape accuracy during training. Experimental results demonstrate that our method produces more accurate and visually coherent head reconstructions with improved expression controllability and detail fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05064v2">A Study of the Framework and Real-World Applications of Language Embedding for 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has rapidly emerged as a transformative technique for real-time 3D scene representation, offering a highly efficient and expressive alternative to Neural Radiance Fields (NeRF). Its ability to render complex scenes with high fidelity has enabled progress across domains such as scene reconstruction, robotics, and interactive content creation. More recently, the integration of Large Language Models (LLMs) and language embeddings into Gaussian Splatting pipelines has opened new possibilities for text-conditioned generation, editing, and semantic scene understanding. Despite these advances, a comprehensive overview of this emerging intersection has been lacking. This survey presents a structured review of current research efforts that combine language guidance with 3D Gaussian Splatting, detailing theoretical foundations, integration strategies, and real-world use cases. We highlight key limitations such as computational bottlenecks, generalizability, and the scarcity of semantically annotated 3D Gaussian data and outline open challenges and future directions for advancing language-guided 3D scene understanding using Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13043v1">IntelliCap: Intelligent Guidance for Consistent View Sampling</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 This work is a pre-print version of a paper that has been accepted to the IEEE International Symposium on Mixed and Augmented Reality for future publication. Project Page: https://mediated-reality.github.io/projects/yasunaga_ismar25/
    </div>
    <details class="paper-abstract">
      Novel view synthesis from images, for example, with 3D Gaussian splatting, has made great progress. Rendering fidelity and speed are now ready even for demanding virtual reality applications. However, the problem of assisting humans in collecting the input images for these rendering algorithms has received much less attention. High-quality view synthesis requires uniform and dense view sampling. Unfortunately, these requirements are not easily addressed by human camera operators, who are in a hurry, impatient, or lack understanding of the scene structure and the photographic process. Existing approaches to guide humans during image acquisition concentrate on single objects or neglect view-dependent material characteristics. We propose a novel situated visualization technique for scanning at multiple scales. During the scanning of a scene, our method identifies important objects that need extended image coverage to properly represent view-dependent appearance. To this end, we leverage semantic segmentation and category identification, ranked by a vision-language model. Spherical proxies are generated around highly ranked objects to guide the user during scanning. Our results show superior performance in real scenes compared to conventional view sampling strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17728v3">Casual3DHDR: Deblurring High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 Accepted to ACM Multimedia 2025. Project page: https://lingzhezhao.github.io/CasualHDRSplat/
    </div>
    <details class="paper-abstract">
      Photo-realistic novel view synthesis from multi-view images, such as neural radiance field (NeRF) and 3D Gaussian Splatting (3DGS), has gained significant attention for its superior performance. However, most existing methods rely on low dynamic range (LDR) images, limiting their ability to capture detailed scenes in high-contrast environments. While some prior works address high dynamic range (HDR) scene reconstruction, they typically require multi-view sharp images with varying exposure times captured at fixed camera positions, which is time-consuming and impractical. To make data acquisition more flexible, we propose \textbf{Casual3DHDR}, a robust one-stage method that reconstructs 3D HDR scenes from casually-captured auto-exposure (AE) videos, even under severe motion blur and unknown, varying exposure times. Our approach integrates a continuous-time camera trajectory into a unified physical imaging model, jointly optimizing exposure times, camera trajectory, and the camera response function (CRF). Extensive experiments on synthetic and real-world datasets demonstrate that \textbf{Casual3DHDR} outperforms existing methods in robustness and rendering quality. Our source code and dataset will be available at https://lingzhezhao.github.io/CasualHDRSplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12720v1">Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 Under review. Project page: https://chenkangjie1123.github.io/Co-Adaptation-3DGS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive performance in novel view synthesis under dense-view settings. However, in sparse-view scenarios, despite the realistic renderings in training views, 3DGS occasionally manifests appearance artifacts in novel views. This paper investigates the appearance artifacts in sparse-view 3DGS and uncovers a core limitation of current approaches: the optimized Gaussians are overly-entangled with one another to aggressively fit the training views, which leads to a neglect of the real appearance distribution of the underlying scene and results in appearance artifacts in novel views. The analysis is based on a proposed metric, termed Co-Adaptation Score (CA), which quantifies the entanglement among Gaussians, i.e., co-adaptation, by computing the pixel-wise variance across multiple renderings of the same viewpoint, with different random subsets of Gaussians. The analysis reveals that the degree of co-adaptation is naturally alleviated as the number of training views increases. Based on the analysis, we propose two lightweight strategies to explicitly mitigate the co-adaptation in sparse-view 3DGS: (1) random gaussian dropout; (2) multiplicative noise injection to the opacity. Both strategies are designed to be plug-and-play, and their effectiveness is validated across various methods and benchmarks. We hope that our insights into the co-adaptation effect will inspire the community to achieve a more comprehensive understanding of sparse-view 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08331v3">SLGaussian: Fast Language Gaussian Splatting in Sparse Views</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 Accepted by ACM MM 2025. Project page: https://chenkangjie1123.github.io/SLGaussian.github.io/
    </div>
    <details class="paper-abstract">
      3D semantic field learning is crucial for applications like autonomous navigation, AR/VR, and robotics, where accurate comprehension of 3D scenes from limited viewpoints is essential. Existing methods struggle under sparse view conditions, relying on inefficient per-scene multi-view optimizations, which are impractical for many real-world tasks. To address this, we propose SLGaussian, a feed-forward method for constructing 3D semantic fields from sparse viewpoints, allowing direct inference of 3DGS-based scenes. By ensuring consistent SAM segmentations through video tracking and using low-dimensional indexing for high-dimensional CLIP features, SLGaussian efficiently embeds language information in 3D space, offering a robust solution for accurate 3D scene understanding under sparse view conditions. In experiments on two-view sparse 3D object querying and segmentation in the LERF and 3D-OVS datasets, SLGaussian outperforms existing methods in chosen IoU, Localization Accuracy, and mIoU. Moreover, our model achieves scene inference in under 30 seconds and open-vocabulary querying in just 0.011 seconds per query.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16392v4">Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 16pages,18figures
    </div>
    <details class="paper-abstract">
      We propose Quadratic Gaussian Splatting (QGS), a novel representation that replaces static primitives with deformable quadric surfaces (e.g., ellipse, paraboloids) to capture intricate geometry. Unlike prior works that rely on Euclidean distance for primitive density modeling--a metric misaligned with surface geometry under deformation--QGS introduces geodesic distance-based density distributions. This innovation ensures that density weights adapt intrinsically to the primitive curvature, preserving consistency during shape changes (e.g., from planar disks to curved paraboloids). By solving geodesic distances in closed form on quadric surfaces, QGS enables surface-aware splatting, where a single primitive can represent complex curvature that previously required dozens of planar surfels, potentially reducing memory usage while maintaining efficient rendering via fast ray-quadric intersection. Experiments on DTU, Tanks and Temples, and MipNeRF360 datasets demonstrate state-of-the-art surface reconstruction, with QGS reducing geometric error (chamfer distance) by 33% over 2DGS and 27% over GOF on the DTU dataset. Crucially, QGS retains competitive appearance quality, bridging the gap between geometric precision and visual fidelity for applications like robotics and immersive reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13287v1">InnerGS: Internal Scenes Rendering via Factorized 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently gained popularity for efficient scene rendering by representing scenes as explicit sets of anisotropic 3D Gaussians. However, most existing work focuses primarily on modeling external surfaces. In this work, we target the reconstruction of internal scenes, which is crucial for applications that require a deep understanding of an object's interior. By directly modeling a continuous volumetric density through the inner 3D Gaussian distribution, our model effectively reconstructs smooth and detailed internal structures from sparse sliced data. Our approach eliminates the need for camera poses, is plug-and-play, and is inherently compatible with any data modalities. We provide cuda implementation at: https://github.com/Shuxin-Liang/InnerGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08124v2">SLAG: Scalable Language-Augmented Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: https://slag-project.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12415v1">TiP4GEN: Text to Immersive Panorama 4D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-17
    </div>
    <details class="paper-abstract">
      With the rapid advancement and widespread adoption of VR/AR technologies, there is a growing demand for the creation of high-quality, immersive dynamic scenes. However, existing generation works predominantly concentrate on the creation of static scenes or narrow perspective-view dynamic scenes, falling short of delivering a truly 360-degree immersive experience from any viewpoint. In this paper, we introduce \textbf{TiP4GEN}, an advanced text-to-dynamic panorama scene generation framework that enables fine-grained content control and synthesizes motion-rich, geometry-consistent panoramic 4D scenes. TiP4GEN integrates panorama video generation and dynamic scene reconstruction to create 360-degree immersive virtual environments. For video generation, we introduce a \textbf{Dual-branch Generation Model} consisting of a panorama branch and a perspective branch, responsible for global and local view generation, respectively. A bidirectional cross-attention mechanism facilitates comprehensive information exchange between the branches. For scene reconstruction, we propose a \textbf{Geometry-aligned Reconstruction Model} based on 3D Gaussian Splatting. By aligning spatial-temporal point clouds using metric depth maps and initializing scene cameras with estimated poses, our method ensures geometric consistency and temporal coherence for the reconstructed scenes. Extensive experiments demonstrate the effectiveness of our proposed designs and the superiority of TiP4GEN in generating visually compelling and motion-coherent dynamic panoramic scenes. Our project page is at https://ke-xing.github.io/TiP4GEN/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12313v1">Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering</a></div>
    <div class="paper-meta">
      📅 2025-08-17
      | 💬 Project page: https://xiaobin2001.github.io/improved-gs-web
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) has achieved impressive performance in real-time rendering, its densification strategy often results in suboptimal reconstruction quality. In this work, we present a comprehensive improvement to the densification pipeline of 3DGS from three perspectives: when to densify, how to densify, and how to mitigate overfitting. Specifically, we propose an Edge-Aware Score to effectively select candidate Gaussians for splitting. We further introduce a Long-Axis Split strategy that reduces geometric distortions introduced by clone and split operations. To address overfitting, we design a set of techniques, including Recovery-Aware Pruning, Multi-step Update, and Growth Control. Our method enhances rendering fidelity without introducing additional training or inference overhead, achieving state-of-the-art performance with fewer Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12015v1">InstDrive: Instance-Aware 3D Gaussian Splatting for Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic driving scenes from dashcam videos has attracted increasing attention due to its significance in autonomous driving and scene understanding. While recent advances have made impressive progress, most methods still unify all background elements into a single representation, hindering both instance-level understanding and flexible scene editing. Some approaches attempt to lift 2D segmentation into 3D space, but often rely on pre-processed instance IDs or complex pipelines to map continuous features to discrete identities. Moreover, these methods are typically designed for indoor scenes with rich viewpoints, making them less applicable to outdoor driving scenarios. In this paper, we present InstDrive, an instance-aware 3D Gaussian Splatting framework tailored for the interactive reconstruction of dynamic driving scene. We use masks generated by SAM as pseudo ground-truth to guide 2D feature learning via contrastive loss and pseudo-supervised objectives. At the 3D level, we introduce regularization to implicitly encode instance identities and enforce consistency through a voxel-based loss. A lightweight static codebook further bridges continuous features and discrete identities without requiring data pre-processing or complex optimization. Quantitative and qualitative experiments demonstrate the effectiveness of InstDrive, and to the best of our knowledge, it is the first framework to achieve 3D instance segmentation in dynamic, open-world driving scenes.More visualizations are available at our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11854v1">ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages</a></div>
    <div class="paper-meta">
      📅 2025-08-16
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce ComplicitSplat, the first attack that exploits standard 3DGS shading methods to create viewpoint-specific camouflage - colors and textures that change with viewing angle - to embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that ComplicitSplat generalizes to successfully attack a variety of popular detector - both single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06587v3">Introducing Unbiased Depth into 2D Gaussian Splatting for High-accuracy Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Accepted to the Journal track of Pacific Graphics 2025
    </div>
    <details class="paper-abstract">
      Recently, 2D Gaussian Splatting (2DGS) has demonstrated superior geometry reconstruction quality than the popular 3DGS by using 2D surfels to approximate thin surfaces. However, it falls short when dealing with glossy surfaces, resulting in visible holes in these areas. We find that the reflection discontinuity causes the issue. To fit the jump from diffuse to specular reflection at different viewing angles, depth bias is introduced in the optimized Gaussian primitives. To address that, we first replace the depth distortion loss in 2DGS with a novel depth convergence loss, which imposes a strong constraint on depth continuity. Then, we rectify the depth criterion in determining the actual surface, which fully accounts for all the intersecting Gaussians along the ray. Qualitative and quantitative evaluations across various datasets reveal that our method significantly improves reconstruction quality, with more complete and accurate surfaces than 2DGS. Code is available at https://github.com/XiaoXinyyx/Unbiased_Surfel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18394v5">Reconstructing Satellites in 3D from Amateur Telescope Images</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Monitoring space objects is crucial for space situational awareness, yet reconstructing 3D satellite models from ground-based telescope images is challenging due to atmospheric turbulence, long observation distances, limited viewpoints, and low signal-to-noise ratios. In this paper, we propose a novel computational imaging framework that overcomes these obstacles by integrating a hybrid image pre-processing pipeline with a joint pose estimation and 3D reconstruction module based on controlled Gaussian Splatting (GS) and Branch-and-Bound (BnB) search. We validate our approach on both synthetic satellite datasets and on-sky observations of China's Tiangong Space Station and the International Space Station, achieving robust 3D reconstructions of low-Earth orbit satellites from ground-based data. Quantitative evaluations using SSIM, PSNR, LPIPS, and Chamfer Distance demonstrate that our method outperforms state-of-the-art NeRF-based approaches, and ablation studies confirm the critical role of each component. Our framework enables high-fidelity 3D satellite monitoring from Earth, offering a cost-effective alternative for space situational awareness. Project page: https://ai4scientificimaging.org/ReconstructingSatellites
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11431v1">Remove360: Benchmarking Residuals After Object Removal in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 arXiv admin note: substantial text overlap with arXiv:2503.17574
    </div>
    <details class="paper-abstract">
      Understanding what semantic information persists after object removal is critical for privacy-preserving 3D reconstruction and editable scene representations. In this work, we introduce a novel benchmark and evaluation framework to measure semantic residuals, the unintended semantic traces left behind, after object removal in 3D Gaussian Splatting. We conduct experiments across a diverse set of indoor and outdoor scenes, showing that current methods can preserve semantic information despite the absence of visual geometry. We also release Remove360, a dataset of pre/post-removal RGB images and object-level masks captured in real-world environments. While prior datasets have focused on isolated object instances, Remove360 covers a broader and more complex range of indoor and outdoor scenes, enabling evaluation of object removal in the context of full-scene representations. Given ground truth images of a scene before and after object removal, we assess whether we can truly eliminate semantic presence, and if downstream models can still infer what was removed. Our findings reveal critical limitations in current 3D object removal techniques and underscore the need for more robust solutions capable of handling real-world complexity. The evaluation framework is available at github.com/spatial-intelligence-ai/Remove360.git. Data are available at huggingface.co/datasets/simkoc/Remove360.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17728v2">Casual3DHDR: High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 Published in ACM Multimedia 2025. Project page: https://lingzhezhao.github.io/CasualHDRSplat/ (Previously titled "CasualHDRSplat: Robust High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos")
    </div>
    <details class="paper-abstract">
      Photo-realistic novel view synthesis from multi-view images, such as neural radiance field (NeRF) and 3D Gaussian Splatting (3DGS), has gained significant attention for its superior performance. However, most existing methods rely on low dynamic range (LDR) images, limiting their ability to capture detailed scenes in high-contrast environments. While some prior works address high dynamic range (HDR) scene reconstruction, they typically require multi-view sharp images with varying exposure times captured at fixed camera positions, which is time-consuming and impractical. To make data acquisition more flexible, we propose \textbf{Casual3DHDR}, a robust one-stage method that reconstructs 3D HDR scenes from casually-captured auto-exposure (AE) videos, even under severe motion blur and unknown, varying exposure times. Our approach integrates a continuous camera trajectory into a unified physical imaging model, jointly optimizing exposure times, camera trajectory, and the camera response function (CRF). Extensive experiments on synthetic and real-world datasets demonstrate that \textbf{Casual3DHDR} outperforms existing methods in robustness and rendering quality. Our source code and dataset will be available at https://lingzhezhao.github.io/CasualHDRSplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05908v2">GBR: Generative Bundle Refinement for High-fidelity Gaussian Splatting with Enhanced Mesh Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Gaussian splatting has gained attention for its efficient representation and rendering of 3D scenes using continuous Gaussian primitives. However, it struggles with sparse-view inputs due to limited geometric and photometric information, causing ambiguities in depth, shape, and texture. we propose GBR: Generative Bundle Refinement, a method for high-fidelity Gaussian splatting and meshing using only 4-6 input views. GBR integrates a neural bundle adjustment module to enhance geometry accuracy and a generative depth refinement module to improve geometry fidelity. More specifically, the neural bundle adjustment module integrates a foundation network to produce initial 3D point maps and point matches from unposed images, followed by bundle adjustment optimization to improve multiview consistency and point cloud accuracy. The generative depth refinement module employs a diffusion-based strategy to enhance geometric details and fidelity while preserving the scale. Finally, for Gaussian splatting optimization, we propose a multimodal loss function incorporating depth and normal consistency, geometric regularization, and pseudo-view supervision, providing robust guidance under sparse-view conditions. Experiments on widely used datasets show that GBR significantly outperforms existing methods under sparse-view inputs. Additionally, GBR demonstrates the ability to reconstruct and render large-scale real-world scenes, such as the Pavilion of Prince Teng and the Great Wall, with remarkable details using only 6 views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11183v1">Versatile Video Tokenization with Generative 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-15
    </div>
    <details class="paper-abstract">
      Video tokenization procedure is critical for a wide range of video processing tasks. Most existing approaches directly transform video into fixed-grid and patch-wise tokens, which exhibit limited versatility. Spatially, uniformly allocating a fixed number of tokens often leads to over-encoding in low-information regions. Temporally, reducing redundancy remains challenging without explicitly distinguishing between static and dynamic content. In this work, we propose the Gaussian Video Transformer (GVT), a versatile video tokenizer built upon a generative 2D Gaussian Splatting (2DGS) strategy. We first extract latent rigid features from a video clip and represent them with a set of 2D Gaussians generated by our proposed Spatio-Temporal Gaussian Embedding (STGE) mechanism in a feed-forward manner. Such generative 2D Gaussians not only enhance spatial adaptability by assigning higher (resp., lower) rendering weights to regions with higher (resp., lower) information content during rasterization, but also improve generalization by avoiding per-video optimization.To enhance the temporal versatility, we introduce a Gaussian Set Partitioning (GSP) strategy that separates the 2D Gaussians into static and dynamic sets, which explicitly model static content shared across different time-steps and dynamic content specific to each time-step, enabling a compact representation.We primarily evaluate GVT on the video reconstruction, while also assessing its performance on action recognition and compression using the UCF101, Kinetics, and DAVIS datasets. Extensive experiments demonstrate that GVT achieves a state-of-the-art video reconstruction quality, outperforms the baseline MAGVIT-v2 in action recognition, and delivers comparable compression performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20469v2">CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D reconstruction techniques and vision-language models have fueled significant progress in 3D semantic understanding, a capability critical to robotics, autonomous driving, and virtual/augmented reality. However, methods that rely on 2D priors are prone to a critical challenge: cross-view semantic inconsistencies induced by occlusion, image blur, and view-dependent variations. These inconsistencies, when propagated via projection supervision, deteriorate the quality of 3D Gaussian semantic fields and introduce artifacts in the rendered outputs. To mitigate this limitation, we propose CCL-LGS, a novel framework that enforces view-consistent semantic supervision by integrating multi-view semantic cues. Specifically, our approach first employs a zero-shot tracker to align a set of SAM-generated 2D masks and reliably identify their corresponding categories. Next, we utilize CLIP to extract robust semantic encodings across views. Finally, our Contrastive Codebook Learning (CCL) module distills discriminative semantic features by enforcing intra-class compactness and inter-class distinctiveness. In contrast to previous methods that directly apply CLIP to imperfect masks, our framework explicitly resolves semantic conflicts while preserving category discriminability. Extensive experiments demonstrate that CCL-LGS outperforms previous state-of-the-art methods. Our project page is available at https://epsilontl.github.io/CCL-LGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10507v1">Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian splatting have significantly improved real-time novel view synthesis, yet insufficient geometric constraints during scene optimization often result in blurred reconstructions of fine-grained details, particularly in regions with high-frequency textures and sharp discontinuities. To address this, we propose a comprehensive optimization framework integrating multisample anti-aliasing (MSAA) with dual geometric constraints. Our system computes pixel colors through adaptive blending of quadruple subsamples, effectively reducing aliasing artifacts in high-frequency components. The framework introduces two constraints: (a) an adaptive weighting strategy that prioritizes under-reconstructed regions through dynamic gradient analysis, and (b) gradient differential constraints enforcing geometric regularization at object boundaries. This targeted optimization enables the model to allocate computational resources preferentially to critical regions requiring refinement while maintaining global consistency. Extensive experimental evaluations across multiple benchmarks demonstrate that our method achieves state-of-the-art performance in detail preservation, particularly in preserving high-frequency textures and sharp discontinuities, while maintaining real-time rendering efficiency. Quantitative metrics and perceptual studies confirm statistically significant improvements over baseline approaches in both structural similarity (SSIM) and perceptual quality (LPIPS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00578v3">Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 CVPR 2025, Project Page: https://speedysplat.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) is a recent 3D scene reconstruction technique that enables real-time rendering of novel views by modeling scenes as parametric point clouds of differentiable 3D Gaussians. However, its rendering speed and model size still present bottlenecks, especially in resource-constrained settings. In this paper, we identify and address two key inefficiencies in 3D-GS to substantially improve rendering speed. These improvements also yield the ancillary benefits of reduced model size and training time. First, we optimize the rendering pipeline to precisely localize Gaussians in the scene, boosting rendering speed without altering visual fidelity. Second, we introduce a novel pruning technique and integrate it into the training pipeline, significantly reducing model size and training time while further raising rendering speed. Our Speedy-Splat approach combines these techniques to accelerate average rendering speed by a drastic $\mathit{6.71\times}$ across scenes from the Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03910v3">DGNS: Deformable Gaussian Splatting and Dynamic Neural Surface for Monocular Dynamic 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction from monocular video is essential for real-world applications. We introduce DGNS, a hybrid framework integrating \underline{D}eformable \underline{G}aussian Splatting and Dynamic \underline{N}eural \underline{S}urfaces, effectively addressing dynamic novel-view synthesis and 3D geometry reconstruction simultaneously. During training, depth maps generated by the deformable Gaussian splatting module guide the ray sampling for faster processing and provide depth supervision within the dynamic neural surface module to improve geometry reconstruction. Conversely, the dynamic neural surface directs the distribution of Gaussian primitives around the surface, enhancing rendering quality. In addition, we propose a depth-filtering approach to further refine depth supervision. Extensive experiments conducted on public datasets demonstrate that DGNS achieves state-of-the-art performance in 3D reconstruction, along with competitive results in novel-view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09977v1">A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 GitHub Repo: https://github.com/heshuting555/Awesome-3DGS-Applications
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a powerful alternative to Neural Radiance Fields (NeRF) for 3D scene representation, offering high-fidelity photorealistic rendering with real-time performance. Beyond novel view synthesis, the explicit and compact nature of 3DGS enables a wide range of downstream applications that require geometric and semantic understanding. This survey provides a comprehensive overview of recent progress in 3DGS applications. It first introduces 2D foundation models that support semantic understanding and control in 3DGS applications, followed by a review of NeRF-based methods that inform their 3DGS counterparts. We then categorize 3DGS applications into segmentation, editing, generation, and other functional tasks. For each, we summarize representative methods, supervision strategies, and learning paradigms, highlighting shared design principles and emerging trends. Commonly used datasets and evaluation protocols are also summarized, along with comparative analyses of recent methods across public benchmarks. To support ongoing research and development, a continually updated repository of papers, code, and resources is maintained at https://github.com/heshuting555/Awesome-3DGS-Applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09912v1">E-4DGS: High-Fidelity Dynamic Reconstruction from the Multi-view Event Cameras</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 16 pages, 10 figures, 5 Tables, accepted by ACMMM 2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis and 4D reconstruction techniques predominantly rely on RGB cameras, thereby inheriting inherent limitations such as the dependence on adequate lighting, susceptibility to motion blur, and a limited dynamic range. Event cameras, offering advantages of low power, high temporal resolution and high dynamic range, have brought a new perspective to addressing the scene reconstruction challenges in high-speed motion and low-light scenes. To this end, we propose E-4DGS, the first event-driven dynamic Gaussian Splatting approach, for novel view synthesis from multi-view event streams with fast-moving cameras. Specifically, we introduce an event-based initialization scheme to ensure stable training and propose event-adaptive slicing splatting for time-aware reconstruction. Additionally, we employ intensity importance pruning to eliminate floating artifacts and enhance 3D consistency, while incorporating an adaptive contrast threshold for more precise optimization. We design a synthetic multi-view camera setup with six moving event cameras surrounding the object in a 360-degree configuration and provide a benchmark multi-view event stream dataset that captures challenging motion scenarios. Our approach outperforms both event-only and event-RGB fusion baselines and paves the way for the exploration of multi-view event-based reconstruction as a novel approach for rapid scene capture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07701v2">Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 This paper has been accepted by IROS 2025. Code: https://github.com/Bistu3DV/MND-GS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves remarkable results in the field of surface reconstruction. However, when Gaussian normal vectors are aligned within the single-view projection plane, while the geometry appears reasonable in the current view, biases may emerge upon switching to nearby views. To address the distance and global matching challenges in multi-view scenes, we design multi-view normal and distance-guided Gaussian splatting. This method achieves geometric depth unification and high-accuracy reconstruction by constraining nearby depth maps and aligning 3D normals. Specifically, for the reconstruction of small indoor and outdoor scenes, we propose a multi-view distance reprojection regularization module that achieves multi-view Gaussian alignment by computing the distance loss between two nearby views and the same Gaussian surface. Additionally, we develop a multi-view normal enhancement module, which ensures consistency across views by matching the normals of pixel points in nearby views and calculating the loss. Extensive experimental results demonstrate that our method outperforms the baseline in both quantitative and qualitative evaluations, significantly enhancing the surface reconstruction capability of 3DGS. Our code will be made publicly available at (https://github.com/Bistu3DV/MND-GS/).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09858v1">HumanGenesis: Agent-Based Geometric and Generative Modeling for Synthetic Human Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-08-13
    </div>
    <details class="paper-abstract">
      \textbf{Synthetic human dynamics} aims to generate photorealistic videos of human subjects performing expressive, intention-driven motions. However, current approaches face two core challenges: (1) \emph{geometric inconsistency} and \emph{coarse reconstruction}, due to limited 3D modeling and detail preservation; and (2) \emph{motion generalization limitations} and \emph{scene inharmonization}, stemming from weak generative capabilities. To address these, we present \textbf{HumanGenesis}, a framework that integrates geometric and generative modeling through four collaborative agents: (1) \textbf{Reconstructor} builds 3D-consistent human-scene representations from monocular video using 3D Gaussian Splatting and deformation decomposition. (2) \textbf{Critique Agent} enhances reconstruction fidelity by identifying and refining poor regions via multi-round MLLM-based reflection. (3) \textbf{Pose Guider} enables motion generalization by generating expressive pose sequences using time-aware parametric encoders. (4) \textbf{Video Harmonizer} synthesizes photorealistic, coherent video via a hybrid rendering pipeline with diffusion, refining the Reconstructor through a Back-to-4D feedback loop. HumanGenesis achieves state-of-the-art performance on tasks including text-guided synthesis, video reenactment, and novel-pose generalization, significantly improving expressiveness, geometric fidelity, and scene integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09855v1">Toward Human-Robot Teaming: Learning Handover Behaviors from 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 3 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Human-robot teaming (HRT) systems often rely on large-scale datasets of human and robot interactions, especially for close-proximity collaboration tasks such as human-robot handovers. Learning robot manipulation policies from raw, real-world image data requires a large number of robot-action trials in the physical environment. Although simulation training offers a cost-effective alternative, the visual domain gap between simulation and robot workspace remains a major limitation. We introduce a method for training HRT policies, focusing on human-to-robot handovers, solely from RGB images without the need for real-robot training or real-robot data collection. The goal is to enable the robot to reliably receive objects from a human with stable grasping while avoiding collisions with the human hand. The proposed policy learner leverages sparse-view Gaussian Splatting reconstruction of human-to-robot handover scenes to generate robot demonstrations containing image-action pairs captured with a camera mounted on the robot gripper. As a result, the simulated camera pose changes in the reconstructed scene can be directly translated into gripper pose changes. Experiments in both Gaussian Splatting reconstructed scene and real-world human-to-robot handover experiments demonstrate that our method serves as a new and effective representation for the human-to-robot handover task, contributing to more seamless and robust HRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09667v1">GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-13
    </div>
    <details class="paper-abstract">
      Reconstructing 3D scenes using 3D Gaussian Splatting (3DGS) from sparse views is an ill-posed problem due to insufficient information, often resulting in noticeable artifacts. While recent approaches have sought to leverage generative priors to complete information for under-constrained regions, they struggle to generate content that remains consistent with input observations. To address this challenge, we propose GSFixer, a novel framework designed to improve the quality of 3DGS representations reconstructed from sparse inputs. The core of our approach is the reference-guided video restoration model, built upon a DiT-based video diffusion model trained on paired artifact 3DGS renders and clean frames with additional reference-based conditions. Considering the input sparse views as references, our model integrates both 2D semantic features and 3D geometric features of reference views extracted from the visual geometry foundation model, enhancing the semantic coherence and 3D consistency when fixing artifact novel views. Furthermore, considering the lack of suitable benchmarks for 3DGS artifact restoration evaluation, we present DL3DV-Res which contains artifact frames rendered using low-quality 3DGS. Extensive experiments demonstrate our GSFixer outperforms current state-of-the-art methods in 3DGS artifact restoration and sparse-view 3D reconstruction. Project page: https://github.com/GVCLab/GSFixer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09610v1">DualPhys-GS: Dual Physically-Guided 3D Gaussian Splatting for Underwater Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In 3D reconstruction of underwater scenes, traditional methods based on atmospheric optical models cannot effectively deal with the selective attenuation of light wavelengths and the effect of suspended particle scattering, which are unique to the water medium, and lead to color distortion, geometric artifacts, and collapsing phenomena at long distances. We propose the DualPhys-GS framework to achieve high-quality underwater reconstruction through a dual-path optimization mechanism. Our approach further develops a dual feature-guided attenuation-scattering modeling mechanism, the RGB-guided attenuation optimization model combines RGB features and depth information and can handle edge and structural details. In contrast, the multi-scale depth-aware scattering model captures scattering effects at different scales using a feature pyramid network and an attention mechanism. Meanwhile, we design several special loss functions. The attenuation scattering consistency loss ensures physical consistency. The water body type adaptive loss dynamically adjusts the weighting coefficients. The edge-aware scattering loss is used to maintain the sharpness of structural edges. The multi-scale feature loss helps to capture global and local structural information. In addition, we design a scene adaptive mechanism that can automatically identify the water-body-type characteristics (e.g., clear coral reef waters or turbid coastal waters) and dynamically adjust the scattering and attenuation parameters and optimization strategies. Experimental results show that our method outperforms existing methods in several metrics, especially in suspended matter-dense regions and long-distance scenes, and the reconstruction quality is significantly improved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23162v2">NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 Project page: https://pku-yuangroup.github.io/NeuralGS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves impressive quality and rendering speed, but with millions of 3D Gaussians and significant storage and transmission costs. In this paper, we aim to develop a simple yet effective method called NeuralGS that compresses the original 3DGS into a compact representation. Our observation is that neural fields like NeRF can represent complex 3D scenes with Multi-Layer Perceptron (MLP) neural networks using only a few megabytes. Thus, NeuralGS effectively adopts the neural field representation to encode the attributes of 3D Gaussians with MLPs, only requiring a small storage size even for a large-scale scene. To achieve this, we adopt a clustering strategy and fit the Gaussians within each cluster using different tiny MLPs, based on importance scores of Gaussians as fitting weights. We experiment on multiple datasets, achieving a 91-times average model size reduction without harming the visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09479v1">SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images</a></div>
    <div class="paper-meta">
      📅 2025-08-13
    </div>
    <details class="paper-abstract">
      Three-dimensional scene reconstruction from sparse-view satellite images is a long-standing and challenging task. While 3D Gaussian Splatting (3DGS) and its variants have recently attracted attention for its high efficiency, existing methods remain unsuitable for satellite images due to incompatibility with rational polynomial coefficient (RPC) models and limited generalization capability. Recent advances in generalizable 3DGS approaches show potential, but they perform poorly on multi-temporal sparse satellite images due to limited geometric constraints, transient objects, and radiometric inconsistencies. To address these limitations, we propose SkySplat, a novel self-supervised framework that integrates the RPC model into the generalizable 3DGS pipeline, enabling more effective use of sparse geometric cues for improved reconstruction. SkySplat relies only on RGB images and radiometric-robust relative height supervision, thereby eliminating the need for ground-truth height maps. Key components include a Cross-Self Consistency Module (CSCM), which mitigates transient object interference via consistency-based masking, and a multi-view consistency aggregation strategy that refines reconstruction results. Compared to per-scene optimization methods, SkySplat achieves an 86 times speedup over EOGS with higher accuracy. It also outperforms generalizable 3DGS baselines, reducing MAE from 13.18 m to 1.80 m on the DFC19 dataset significantly, and demonstrates strong cross-dataset generalization on the MVS3D benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18764v2">Efficient Differentiable Hardware Rasterization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 8 pages,2 figures
    </div>
    <details class="paper-abstract">
      Recent works demonstrate the advantages of hardware rasterization for 3D Gaussian Splatting (3DGS) in forward-pass rendering through fast GPU-optimized graphics and fixed memory footprint. However, extending these benefits to backward-pass gradient computation remains challenging due to graphics pipeline constraints. We present a differentiable hardware rasterizer for 3DGS that overcomes the memory and performance limitations of tile-based software rasterization. Our solution employs programmable blending for per-pixel gradient computation combined with a hybrid gradient reduction strategy (quad-level + subgroup) in fragment shaders, achieving over 10x faster backward rasterization versus naive atomic operations and 3x speedup over the canonical tile-based rasterizer. Systematic evaluation reveals 16-bit render targets (float16 and unorm16) as the optimal accuracy-efficiency trade-off, achieving higher gradient accuracy among mixed-precision rendering formats with execution speeds second only to unorm8, while float32 texture incurs severe forward pass performance degradation due to suboptimal hardware optimizations. Our method with float16 formats demonstrates 3.07x acceleration in full pipeline execution (forward + backward passes) on RTX4080 GPUs with the MipNeRF 360 dataset, outperforming the baseline tile-based renderer while preserving hardware rasterization's memory efficiency advantages -- incurring merely 2.67% of the memory overhead required for splat sorting operations. This work presents a unified differentiable hardware rasterization method that simultaneously optimizes runtime and memory usage for 3DGS, making it particularly suitable for resource-constrained devices with limited memory capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01367v2">3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10227v1">EntropyGS: An Efficient Entropy Coding on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-13
    </div>
    <details class="paper-abstract">
      As an emerging novel view synthesis approach, 3D Gaussian Splatting (3DGS) demonstrates fast training/rendering with superior visual quality. The two tasks of 3DGS, Gaussian creation and view rendering, are typically separated over time or devices, and thus storage/transmission and finally compression of 3DGS Gaussians become necessary. We begin with a correlation and statistical analysis of 3DGS Gaussian attributes. An inspiring finding in this work reveals that spherical harmonic AC attributes precisely follow Laplace distributions, while mixtures of Gaussian distributions can approximate rotation, scaling, and opacity. Additionally, harmonic AC attributes manifest weak correlations with other attributes except for inherited correlations from a color space. A factorized and parameterized entropy coding method, EntropyGS, is hereinafter proposed. During encoding, distribution parameters of each Gaussian attribute are estimated to assist their entropy coding. The quantization for entropy coding is adaptively performed according to Gaussian attribute types. EntropyGS demonstrates about 30x rate reduction on benchmark datasets while maintaining similar rendering quality compared to input 3DGS data, with a fast encoding and decoding time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09068v1">A new dataset and comparison for multi-camera frame synthesis</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 SPIE2025 - Applications of Digital Image Processing XLVIII accepted manuscript
    </div>
    <details class="paper-abstract">
      Many methods exist for frame synthesis in image sequences but can be broadly categorised into frame interpolation and view synthesis techniques. Fundamentally, both frame interpolation and view synthesis tackle the same task, interpolating a frame given surrounding frames in time or space. However, most frame interpolation datasets focus on temporal aspects with single cameras moving through time and space, while view synthesis datasets are typically biased toward stereoscopic depth estimation use cases. This makes direct comparison between view synthesis and frame interpolation methods challenging. In this paper, we develop a novel multi-camera dataset using a custom-built dense linear camera array to enable fair comparison between these approaches. We evaluate classical and deep learning frame interpolators against a view synthesis method (3D Gaussian Splatting) for the task of view in-betweening. Our results reveal that deep learning methods do not significantly outperform classical methods on real image data, with 3D Gaussian Splatting actually underperforming frame interpolators by as much as 3.5 dB PSNR. However, in synthetic scenes, the situation reverses -- 3D Gaussian Splatting outperforms frame interpolation algorithms by almost 5 dB PSNR at a 95% confidence level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08867v1">GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Novel view synthesis with neural models has advanced rapidly in recent years, yet adapting these models to scene changes remains an open problem. Existing methods are either labor-intensive, requiring extensive model retraining, or fail to capture detailed types of changes over time. In this paper, we present GaussianUpdate, a novel approach that combines 3D Gaussian representation with continual learning to address these challenges. Our method effectively updates the Gaussian radiance fields with current data while preserving information from past scenes. Unlike existing methods, GaussianUpdate explicitly models different types of changes through a novel multi-stage update strategy. Additionally, we introduce a visibility-aware continual learning approach with generative replay, enabling self-aware updating without the need to store images. The experiments on the benchmark dataset demonstrate our method achieves superior and real-time rendering with the capability of visualizing changes over different times
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06424v2">Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Recent 4D reconstruction methods have yielded impressive results but rely on sharp videos as supervision. However, motion blur often occurs in videos due to camera shake and object movement, while existing methods render blurry results when using such videos for reconstructing 4D models. Although a few approaches attempted to address the problem, they struggled to produce high-quality results, due to the inaccuracy in estimating continuous dynamic representations within the exposure time. Encouraged by recent works in 3D motion trajectory modeling using 3D Gaussian Splatting (3DGS), we take 3DGS as the scene representation manner, and propose Deblur4DGS to reconstruct a high-quality 4D model from blurry monocular video. Specifically, we transform continuous dynamic representations estimation within an exposure time into the exposure time estimation. Moreover, we introduce the exposure regularization term, multi-frame, and multi-resolution consistency regularization term to avoid trivial solutions. Furthermore, to better represent objects with large motion, we suggest blur-aware variable canonical Gaussians. Beyond novel-view synthesis, Deblur4DGS can be applied to improve blurry video from multiple perspectives, including deblurring, frame interpolation, and video stabilization. Extensive experiments in both synthetic and real-world data on the above four tasks show that Deblur4DGS outperforms state-of-the-art 4D reconstruction methods. The codes are available at https://github.com/ZcsrenlongZ/Deblur4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08624v1">Communication Efficient Robotic Mixed Reality with Gaussian Splatting Cross-Layer Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 14 pages, 18 figures, to appear in IEEE Transactions on Cognitive Communications and Networking
    </div>
    <details class="paper-abstract">
      Realizing low-cost communication in robotic mixed reality (RoboMR) systems presents a challenge, due to the necessity of uploading high-resolution images through wireless channels. This paper proposes Gaussian splatting (GS) RoboMR (GSMR), which enables the simulator to opportunistically render a photo-realistic view from the robot's pose by calling ``memory'' from a GS model, thus reducing the need for excessive image uploads. However, the GS model may involve discrepancies compared to the actual environments. To this end, a GS cross-layer optimization (GSCLO) framework is further proposed, which jointly optimizes content switching (i.e., deciding whether to upload image or not) and power allocation (i.e., adjusting to content profiles) across different frames by minimizing a newly derived GSMR loss function. The GSCLO problem is addressed by an accelerated penalty optimization (APO) algorithm that reduces computational complexity by over $10$x compared to traditional branch-and-bound and search algorithms. Moreover, variants of GSCLO are presented to achieve robust, low-power, and multi-robot GSMR. Extensive experiments demonstrate that the proposed GSMR paradigm and GSCLO method achieve significant improvements over existing benchmarks on both wheeled and legged robots in terms of diverse metrics in various scenarios. For the first time, it is found that RoboMR can be achieved with ultra-low communication costs, and mixture of data is useful for enhancing GS performance in dynamic scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09239v1">Gradient-Direction-Aware Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      The emergence of 3D Gaussian Splatting (3DGS) has significantly advanced novel view synthesis through explicit scene representation, enabling real-time photorealistic rendering. However, existing approaches manifest two critical limitations in complex scenarios: (1) Over-reconstruction occurs when persistent large Gaussians cannot meet adaptive splitting thresholds during density control. This is exacerbated by conflicting gradient directions that prevent effective splitting of these Gaussians; (2) Over-densification of Gaussians occurs in regions with aligned gradient aggregation, leading to redundant component proliferation. This redundancy significantly increases memory overhead due to unnecessary data retention. We present Gradient-Direction-Aware Gaussian Splatting (GDAGS), a gradient-direction-aware adaptive density control framework to address these challenges. Our key innovations: the gradient coherence ratio (GCR), computed through normalized gradient vector norms, which explicitly discriminates Gaussians with concordant versus conflicting gradient directions; and a nonlinear dynamic weighting mechanism leverages the GCR to enable gradient-direction-aware density control. Specifically, GDAGS prioritizes conflicting-gradient Gaussians during splitting operations to enhance geometric details while suppressing redundant concordant-direction Gaussians. Conversely, in cloning processes, GDAGS promotes concordant-direction Gaussian densification for structural completion while preventing conflicting-direction Gaussian overpopulation. Comprehensive evaluations across diverse real-world benchmarks demonstrate that GDAGS achieves superior rendering quality while effectively mitigating over-reconstruction, suppressing over-densification, and constructing compact scene representations with 50\% reduced memory consumption through optimized Gaussians utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10936v1">Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Collaborative perception enables connected vehicles to share information, overcoming occlusions and extending the limited sensing range inherent in single-agent (non-collaborative) systems. Existing vision-only methods for 3D semantic occupancy prediction commonly rely on dense 3D voxels, which incur high communication costs, or 2D planar features, which require accurate depth estimation or additional supervision, limiting their applicability to collaborative scenarios. To address these challenges, we propose the first approach leveraging sparse 3D semantic Gaussian splatting for collaborative 3D semantic occupancy prediction. By sharing and fusing intermediate Gaussian primitives, our method provides three benefits: a neighborhood-based cross-agent fusion that removes duplicates and suppresses noisy or inconsistent Gaussians; a joint encoding of geometry and semantics in each primitive, which reduces reliance on depth supervision and allows simple rigid alignment; and sparse, object-centric messages that preserve structural information while reducing communication volume. Extensive experiments demonstrate that our approach outperforms single-agent perception and baseline collaborative methods by +8.42 and +3.28 points in mIoU, and +5.11 and +22.41 points in IoU, respectively. When further reducing the number of transmitted Gaussians, our method still achieves a +1.9 improvement in mIoU, using only 34.6% communication volume, highlighting robust performance under limited communication budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08252v1">ReferSplat: Referring Segmentation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 ICML 2025 Oral, Code: https://github.com/heshuting555/ReferSplat
    </div>
    <details class="paper-abstract">
      We introduce Referring 3D Gaussian Splatting Segmentation (R3DGS), a new task that aims to segment target objects in a 3D Gaussian scene based on natural language descriptions, which often contain spatial relationships or object attributes. This task requires the model to identify newly described objects that may be occluded or not directly visible in a novel view, posing a significant challenge for 3D multi-modal understanding. Developing this capability is crucial for advancing embodied AI. To support research in this area, we construct the first R3DGS dataset, Ref-LERF. Our analysis reveals that 3D multi-modal understanding and spatial relationship modeling are key challenges for R3DGS. To address these challenges, we propose ReferSplat, a framework that explicitly models 3D Gaussian points with natural language expressions in a spatially aware paradigm. ReferSplat achieves state-of-the-art performance on both the newly proposed R3DGS task and 3D open-vocabulary segmentation benchmarks. Dataset and code are available at https://github.com/heshuting555/ReferSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08219v1">SAGOnline: Segment Any Gaussians Online</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 19 pages, 10 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful paradigm for explicit 3D scene representation, yet achieving efficient and consistent 3D segmentation remains challenging. Current methods suffer from prohibitive computational costs, limited 3D spatial reasoning, and an inability to track multiple objects simultaneously. We present Segment Any Gaussians Online (SAGOnline), a lightweight and zero-shot framework for real-time 3D segmentation in Gaussian scenes that addresses these limitations through two key innovations: (1) a decoupled strategy that integrates video foundation models (e.g., SAM2) for view-consistent 2D mask propagation across synthesized views; and (2) a GPU-accelerated 3D mask generation and Gaussian-level instance labeling algorithm that assigns unique identifiers to 3D primitives, enabling lossless multi-object tracking and segmentation across views. SAGOnline achieves state-of-the-art performance on NVOS (92.7% mIoU) and Spin-NeRF (95.2% mIoU) benchmarks, outperforming Feature3DGS, OmniSeg3D-gs, and SA3D by 15--1500 times in inference speed (27 ms/frame). Qualitative results demonstrate robust multi-object segmentation and tracking in complex scenes. Our contributions include: (i) a lightweight and zero-shot framework for 3D segmentation in Gaussian scenes, (ii) explicit labeling of Gaussian primitives enabling simultaneous segmentation and tracking, and (iii) the effective adaptation of 2D video foundation models to the 3D domain. This work allows real-time rendering and 3D scene understanding, paving the way for practical AR/VR and robotic applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08136v1">FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      The success of 3DGS in generative and editing applications has sparked growing interest in 3DGS-based style transfer. However, current methods still face two major challenges: (1) multi-view inconsistency often leads to style conflicts, resulting in appearance smoothing and distortion; and (2) heavy reliance on VGG features, which struggle to disentangle style and content from style images, often causing content leakage and excessive stylization. To tackle these issues, we introduce \textbf{FantasyStyle}, a 3DGS-based style transfer framework, and the first to rely entirely on diffusion model distillation. It comprises two key components: (1) \textbf{Multi-View Frequency Consistency}. We enhance cross-view consistency by applying a 3D filter to multi-view noisy latent, selectively reducing low-frequency components to mitigate stylized prior conflicts. (2) \textbf{Controllable Stylized Distillation}. To suppress content leakage from style images, we introduce negative guidance to exclude undesired content. In addition, we identify the limitations of Score Distillation Sampling and Delta Denoising Score in 3D style transfer and remove the reconstruction term accordingly. Building on these insights, we propose a controllable stylized distillation that leverages negative guidance to more effectively optimize the 3D Gaussians. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art approaches, achieving higher stylization quality and visual realism across various scenes and styles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07897v1">NeeCo: Image Synthesis of Novel Instrument States Based on Dynamic and Deformable 3D Gaussian Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Computer vision-based technologies significantly enhance surgical automation by advancing tool tracking, detection, and localization. However, Current data-driven approaches are data-voracious, requiring large, high-quality labeled image datasets, which limits their application in surgical data science. Our Work introduces a novel dynamic Gaussian Splatting technique to address the data scarcity in surgical image datasets. We propose a dynamic Gaussian model to represent dynamic surgical scenes, enabling the rendering of surgical instruments from unseen viewpoints and deformations with real tissue backgrounds. We utilize a dynamic training adjustment strategy to address challenges posed by poorly calibrated camera poses from real-world scenarios. Additionally, we propose a method based on dynamic Gaussians for automatically generating annotations for our synthetic data. For evaluation, we constructed a new dataset featuring seven scenes with 14,000 frames of tool and camera motion and tool jaw articulation, with a background of an ex-vivo porcine model. Using this dataset, we synthetically replicate the scene deformation from the ground truth data, allowing direct comparisons of synthetic image quality. Experimental results illustrate that our method generates photo-realistic labeled image datasets with the highest values in Peak-Signal-to-Noise Ratio (29.87). We further evaluate the performance of medical-specific neural networks trained on real and synthetic images using an unseen real-world image dataset. Our results show that the performance of models trained on synthetic images generated by the proposed method outperforms those trained with state-of-the-art standard data augmentation by 10%, leading to an overall improvement in model performances by nearly 15%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17288v3">GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Occupancy estimation has become a prominent task in 3D computer vision, particularly within the autonomous driving community. In this paper, we present a novel approach to occupancy estimation, termed GaussianFlowOcc, which is inspired by Gaussian Splatting and replaces traditional dense voxel grids with a sparse 3D Gaussian representation. Our efficient model architecture based on a Gaussian Transformer significantly reduces computational and memory requirements by eliminating the need for expensive 3D convolutions used with inefficient voxel-based representations that predominantly represent empty 3D spaces. GaussianFlowOcc effectively captures scene dynamics by estimating temporal flow for each Gaussian during the overall network training process, offering a straightforward solution to a complex problem that is often neglected by existing methods. Moreover, GaussianFlowOcc is designed for scalability, as it employs weak supervision and does not require costly dense 3D voxel annotations based on additional data (e.g., LiDAR). Through extensive experimentation, we demonstrate that GaussianFlowOcc significantly outperforms all previous methods for weakly supervised occupancy estimation on the nuScenes dataset while featuring an inference speed that is 50 times faster than current SOTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07701v1">Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 This paper has been accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves remarkable results in the field of surface reconstruction. However, when Gaussian normal vectors are aligned within the single-view projection plane, while the geometry appears reasonable in the current view, biases may emerge upon switching to nearby views. To address the distance and global matching challenges in multi-view scenes, we design multi-view normal and distance-guided Gaussian splatting. This method achieves geometric depth unification and high-accuracy reconstruction by constraining nearby depth maps and aligning 3D normals. Specifically, for the reconstruction of small indoor and outdoor scenes, we propose a multi-view distance reprojection regularization module that achieves multi-view Gaussian alignment by computing the distance loss between two nearby views and the same Gaussian surface. Additionally, we develop a multi-view normal enhancement module, which ensures consistency across views by matching the normals of pixel points in nearby views and calculating the loss. Extensive experimental results demonstrate that our method outperforms the baseline in both quantitative and qualitative evaluations, significantly enhancing the surface reconstruction capability of 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03643v3">Uni3R: Unified 3D Reconstruction and Semantic Understanding via Generalizable Gaussian Splatting from Unposed Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 The code is available at https://github.com/HorizonRobotics/Uni3R
    </div>
    <details class="paper-abstract">
      Reconstructing and semantically interpreting 3D scenes from sparse 2D views remains a fundamental challenge in computer vision. Conventional methods often decouple semantic understanding from reconstruction or necessitate costly per-scene optimization, thereby restricting their scalability and generalizability. In this paper, we introduce Uni3R, a novel feed-forward framework that jointly reconstructs a unified 3D scene representation enriched with open-vocabulary semantics, directly from unposed multi-view images. Our approach leverages a Cross-View Transformer to robustly integrate information across arbitrary multi-view inputs, which then regresses a set of 3D Gaussian primitives endowed with semantic feature fields. This unified representation facilitates high-fidelity novel view synthesis, open-vocabulary 3D semantic segmentation, and depth prediction, all within a single, feed-forward pass. Extensive experiments demonstrate that Uni3R establishes a new state-of-the-art across multiple benchmarks, including 25.07 PSNR on RE10K and 55.84 mIoU on ScanNet. Our work signifies a novel paradigm towards generalizable, unified 3D scene reconstruction and understanding. The code is available at https://github.com/HorizonRobotics/Uni3R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16392v3">Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 16pages,18figures
    </div>
    <details class="paper-abstract">
      We propose Quadratic Gaussian Splatting (QGS), a novel representation that replaces static primitives with deformable quadric surfaces (e.g., ellipse, paraboloids) to capture intricate geometry. Unlike prior works that rely on Euclidean distance for primitive density modeling--a metric misaligned with surface geometry under deformation--QGS introduces geodesic distance-based density distributions. This innovation ensures that density weights adapt intrinsically to the primitive curvature, preserving consistency during shape changes (e.g., from planar disks to curved paraboloids). By solving geodesic distances in closed form on quadric surfaces, QGS enables surface-aware splatting, where a single primitive can represent complex curvature that previously required dozens of planar surfels, potentially reducing memory usage while maintaining efficient rendering via fast ray-quadric intersection. Experiments on DTU, Tanks and Temples, and MipNeRF360 datasets demonstrate state-of-the-art surface reconstruction, with QGS reducing geometric error (chamfer distance) by 33% over 2DGS and 27% over GOF on the DTU dataset. Crucially, QGS retains competitive appearance quality, bridging the gap between geometric precision and visual fidelity for applications like robotics and immersive reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07557v1">Splat4D: Diffusion-Enhanced 4D Gaussian Splatting for Temporally and Spatially Consistent Content Creation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Generating high-quality 4D content from monocular videos for applications such as digital humans and AR/VR poses challenges in ensuring temporal and spatial consistency, preserving intricate details, and incorporating user guidance effectively. To overcome these challenges, we introduce Splat4D, a novel framework enabling high-fidelity 4D content generation from a monocular video. Splat4D achieves superior performance while maintaining faithful spatial-temporal coherence by leveraging multi-view rendering, inconsistency identification, a video diffusion model, and an asymmetric U-Net for refinement. Through extensive evaluations on public benchmarks, Splat4D consistently demonstrates state-of-the-art performance across various metrics, underscoring the efficacy of our approach. Additionally, the versatility of Splat4D is validated in various applications such as text/image conditioned 4D generation, 4D human generation, and text-guided content editing, producing coherent outcomes following user instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03310v2">3D Gaussian Splatting Data Compression with Mixture of Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) data compression is crucial for enabling efficient storage and transmission in 3D scene modeling. However, its development remains limited due to inadequate entropy models and suboptimal quantization strategies for both lossless and lossy compression scenarios, where existing methods have yet to 1) fully leverage hyperprior information to construct robust conditional entropy models, and 2) apply fine-grained, element-wise quantization strategies for improved compression granularity. In this work, we propose a novel Mixture of Priors (MoP) strategy to simultaneously address these two challenges. Specifically, inspired by the Mixture-of-Experts (MoE) paradigm, our MoP approach processes hyperprior information through multiple lightweight MLPs to generate diverse prior features, which are subsequently integrated into the MoP feature via a gating mechanism. To enhance lossless compression, the resulting MoP feature is utilized as a hyperprior to improve conditional entropy modeling. Meanwhile, for lossy compression, we employ the MoP feature as guidance information in an element-wise quantization procedure, leveraging a prior-guided Coarse-to-Fine Quantization (C2FQ) strategy with a predefined quantization step value. Specifically, we expand the quantization step value into a matrix and adaptively refine it from coarse to fine granularity, guided by the MoP feature, thereby obtaining a quantization step matrix that facilitates element-wise quantization. Extensive experiments demonstrate that our proposed 3DGS data compression framework achieves state-of-the-art performance across multiple benchmarks, including Mip-NeRF360, BungeeNeRF, DeepBlending, and Tank&Temples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07483v1">Novel View Synthesis with Gaussian Splatting: Impact on Photogrammetry Model Accuracy and Resolution</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      In this paper, I present a comprehensive study comparing Photogrammetry and Gaussian Splatting techniques for 3D model reconstruction and view synthesis. I created a dataset of images from a real-world scene and constructed 3D models using both methods. To evaluate the performance, I compared the models using structural similarity index (SSIM), peak signal-to-noise ratio (PSNR), learned perceptual image patch similarity (LPIPS), and lp/mm resolution based on the USAF resolution chart. A significant contribution of this work is the development of a modified Gaussian Splatting repository, which I forked and enhanced to enable rendering images from novel camera poses generated in the Blender environment. This innovation allows for the synthesis of high-quality novel views, showcasing the flexibility and potential of Gaussian Splatting. My investigation extends to an augmented dataset that includes both original ground images and novel views synthesized via Gaussian Splatting. This augmented dataset was employed to generate a new photogrammetry model, which was then compared against the original photogrammetry model created using only the original images. The results demonstrate the efficacy of using Gaussian Splatting to generate novel high-quality views and its potential to improve photogrammetry-based 3D reconstructions. The comparative analysis highlights the strengths and limitations of both approaches, providing valuable information for applications in extended reality (XR), photogrammetry, and autonomous vehicle simulations. Code is available at https://github.com/pranavc2255/gaussian-splatting-novel-view-render.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2210.00379v7">NeRF: Neural Radiance Field in 3D Vision: A Comprehensive Review (Updated Post-Gaussian Splatting)</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Updated Post-Gaussian Splatting
    </div>
    <details class="paper-abstract">
      In March 2020, Neural Radiance Field (NeRF) revolutionized Computer Vision, allowing for implicit, neural network-based scene representation and novel view synthesis. NeRF models have found diverse applications in robotics, urban mapping, autonomous navigation, virtual reality/augmented reality, and more. In August 2023, Gaussian Splatting, a direct competitor to the NeRF-based framework, was proposed, gaining tremendous momentum and overtaking NeRF-based research in terms of interest as the dominant framework for novel view synthesis. We present a comprehensive survey of NeRF papers from the past five years (2020-2025). These include papers from the pre-Gaussian Splatting era, where NeRF dominated the field for novel view synthesis and 3D implicit and hybrid representation neural field learning. We also include works from the post-Gaussian Splatting era where NeRF and implicit/hybrid neural fields found more niche applications. Our survey is organized into architecture and application-based taxonomies in the pre-Gaussian Splatting era, as well as a categorization of active research areas for NeRF, neural field, and implicit/hybrid neural representation methods. We provide an introduction to the theory of NeRF and its training via differentiable volume rendering. We also present a benchmark comparison of the performance and speed of classical NeRF, implicit and hybrid neural representation, and neural field models, and an overview of key datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07409v1">CharacterShot: Controllable and Consistent 4D Character Animation</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 13 pages, 10 figures. Code at https://github.com/Jeoyal/CharacterShot
    </div>
    <details class="paper-abstract">
      In this paper, we propose \textbf{CharacterShot}, a controllable and consistent 4D character animation framework that enables any individual designer to create dynamic 3D characters (i.e., 4D character animation) from a single reference character image and a 2D pose sequence. We begin by pretraining a powerful 2D character animation model based on a cutting-edge DiT-based image-to-video model, which allows for any 2D pose sequnce as controllable signal. We then lift the animation model from 2D to 3D through introducing dual-attention module together with camera prior to generate multi-view videos with spatial-temporal and spatial-view consistency. Finally, we employ a novel neighbor-constrained 4D gaussian splatting optimization on these multi-view videos, resulting in continuous and stable 4D character representations. Moreover, to improve character-centric performance, we construct a large-scale dataset Character4D, containing 13,115 unique characters with diverse appearances and motions, rendered from multiple viewpoints. Extensive experiments on our newly constructed benchmark, CharacterBench, demonstrate that our approach outperforms current state-of-the-art methods. Code, models, and datasets will be publicly available at https://github.com/Jeoyal/CharacterShot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24009v2">Learning 3D-Gaussian Simulators from RGB Videos</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Realistic simulation is critical for applications ranging from robotics to animation. Learned simulators have emerged as a possibility to capture real world physics directly from video data, but very often require privileged information such as depth information, particle tracks and hand-engineered features to maintain spatial and temporal consistency. These strong inductive biases or ground truth 3D information help in domains where data is sparse but limit scalability and generalization in data rich regimes. To overcome the key limitations, we propose 3DGSim, a learned 3D simulator that directly learns physical interactions from multi-view RGB videos. 3DGSim unifies 3D scene reconstruction, particle dynamics prediction and video synthesis into an end-to-end trained framework. It adopts MVSplat to learn a latent particle-based representation of 3D scenes, a Point Transformer for particle dynamics, a Temporal Merging module for consistent temporal aggregation and Gaussian Splatting to produce novel view renderings. By jointly training inverse rendering and dynamics forecasting, 3DGSim embeds the physical properties into point-wise latent features. This enables the model to capture diverse physical behaviors, from rigid to elastic, cloth-like dynamics, and boundary conditions (e.g. fixed cloth corner), along with realistic lighting effects that also generalize to unseen multibody interactions and novel scene edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07372v1">DIP-GS: Deep Image Prior For Gaussian Splatting Sparse View Recovery</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a leading 3D scene reconstruction method, obtaining high-quality reconstruction with real-time rendering runtime performance. The main idea behind 3DGS is to represent the scene as a collection of 3D gaussians, while learning their parameters to fit the given views of the scene. While achieving superior performance in the presence of many views, 3DGS struggles with sparse view reconstruction, where the input views are sparse and do not fully cover the scene and have low overlaps. In this paper, we propose DIP-GS, a Deep Image Prior (DIP) 3DGS representation. By using the DIP prior, which utilizes internal structure and patterns, with coarse-to-fine manner, DIP-based 3DGS can operate in scenarios where vanilla 3DGS fails, such as sparse view recovery. Note that our approach does not use any pre-trained models such as generative models and depth estimation, but rather relies only on the input frames. Among such methods, DIP-GS obtains state-of-the-art (SOTA) competitive results on various sparse-view reconstruction tasks, demonstrating its capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07355v1">GS4Buildings: Prior-Guided Gaussian Splatting for 3D Building Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Accepted for presentation at ISPRS 3D GeoInfo & Smart Data, Smart Cities 2025, Kashiwa, Japan. To appear in the ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences
    </div>
    <details class="paper-abstract">
      Recent advances in Gaussian Splatting (GS) have demonstrated its effectiveness in photo-realistic rendering and 3D reconstruction. Among these, 2D Gaussian Splatting (2DGS) is particularly suitable for surface reconstruction due to its flattened Gaussian representation and integrated normal regularization. However, its performance often degrades in large-scale and complex urban scenes with frequent occlusions, leading to incomplete building reconstructions. We propose GS4Buildings, a novel prior-guided Gaussian Splatting method leveraging the ubiquity of semantic 3D building models for robust and scalable building surface reconstruction. Instead of relying on traditional Structure-from-Motion (SfM) pipelines, GS4Buildings initializes Gaussians directly from low-level Level of Detail (LoD)2 semantic 3D building models. Moreover, we generate prior depth and normal maps from the planar building geometry and incorporate them into the optimization process, providing strong geometric guidance for surface consistency and structural accuracy. We also introduce an optional building-focused mode that limits reconstruction to building regions, achieving a 71.8% reduction in Gaussian primitives and enabling a more efficient and compact representation. Experiments on urban datasets demonstrate that GS4Buildings improves reconstruction completeness by 20.5% and geometric accuracy by 32.8%. These results highlight the potential of semantic building model integration to advance GS-based reconstruction toward real-world urban applications such as smart cities and digital twins. Our project is available: https://github.com/zqlin0521/GS4Buildings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05591v2">QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 ICCV 2025. Project page: https://liu115.github.io/quicksplat, Video: https://youtu.be/2IA_gnFvFG8
    </div>
    <details class="paper-abstract">
      Surface reconstruction is fundamental to computer vision and graphics, enabling applications in 3D modeling, mixed reality, robotics, and more. Existing approaches based on volumetric rendering obtain promising results, but optimize on a per-scene basis, resulting in a slow optimization that can struggle to model under-observed or textureless regions. We introduce QuickSplat, which learns data-driven priors to generate dense initializations for 2D gaussian splatting optimization of large-scale indoor scenes. This provides a strong starting point for the reconstruction, which accelerates the convergence of the optimization and improves the geometry of flat wall structures. We further learn to jointly estimate the densification and update of the scene parameters during each iteration; our proposed densifier network predicts new Gaussians based on the rendering gradients of existing ones, removing the needs of heuristics for densification. Extensive experiments on large-scale indoor scene reconstruction demonstrate the superiority of our data-driven optimization. Concretely, we accelerate runtime by 8x, while decreasing depth errors by up to 48% in comparison to state of the art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07263v1">Fading the Digital Ink: A Universal Black-Box Attack Framework for 3DGS Watermarking Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      With the rise of 3D Gaussian Splatting (3DGS), a variety of digital watermarking techniques, embedding either 1D bitstreams or 2D images, are used for copyright protection. However, the robustness of these watermarking techniques against potential attacks remains underexplored. This paper introduces the first universal black-box attack framework, the Group-based Multi-objective Evolutionary Attack (GMEA), designed to challenge these watermarking systems. We formulate the attack as a large-scale multi-objective optimization problem, balancing watermark removal with visual quality. In a black-box setting, we introduce an indirect objective function that blinds the watermark detector by minimizing the standard deviation of features extracted by a convolutional network, thus rendering the feature maps uninformative. To manage the vast search space of 3DGS models, we employ a group-based optimization strategy to partition the model into multiple, independent sub-optimization problems. Experiments demonstrate that our framework effectively removes both 1D and 2D watermarks from mainstream 3DGS watermarking methods while maintaining high visual fidelity. This work reveals critical vulnerabilities in existing 3DGS copyright protection schemes and calls for the development of more robust watermarking systems.
    </details>
</div>
