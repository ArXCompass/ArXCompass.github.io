# gaussian splatting - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20291v1">RL-GSBridge: 3D Gaussian Splatting Based Real2Sim2Real Method for Robotic Manipulation Learning</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 7 pages, 5 figures, 4 tables, under review by ICRA2025
    </div>
    <details class="paper-abstract">
      Sim-to-Real refers to the process of transferring policies learned in simulation to the real world, which is crucial for achieving practical robotics applications. However, recent Sim2real methods either rely on a large amount of augmented data or large learning models, which is inefficient for specific tasks. In recent years, radiance field-based reconstruction methods, especially the emergence of 3D Gaussian Splatting, making it possible to reproduce realistic real-world scenarios. To this end, we propose a novel real-to-sim-to-real reinforcement learning framework, RL-GSBridge, which introduces a mesh-based 3D Gaussian Splatting method to realize zero-shot sim-to-real transfer for vision-based deep reinforcement learning. We improve the mesh-based 3D GS modeling method by using soft binding constraints, enhancing the rendering quality of mesh models. We then employ a GS editing approach to synchronize rendering with the physics simulator, reflecting the interactions of the physical robot more accurately. Through a series of sim-to-real robotic arm experiments, including grasping and pick-and-place tasks, we demonstrate that RL-GSBridge maintains a satisfactory success rate in real-world task completion during sim-to-real transfer. Furthermore, a series of rendering metrics and visualization results indicate that our proposed mesh-based 3D Gaussian reduces artifacts in unstructured objects, demonstrating more realistic rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15518v2">Feature Splatting for Better Novel View Synthesis with Low Overlap</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as a very promising scene representation, achieving state-of-the-art quality in novel view synthesis significantly faster than competing alternatives. However, its use of spherical harmonics to represent scene colors limits the expressivity of 3D Gaussians and, as a consequence, the capability of the representation to generalize as we move away from the training views. In this paper, we propose to encode the color information of 3D Gaussians into per-Gaussian feature vectors, which we denote as Feature Splatting (FeatSplat). To synthesize a novel view, Gaussians are first "splatted" into the image plane, then the corresponding feature vectors are alpha-blended, and finally the blended vector is decoded by a small MLP to render the RGB pixel values. To further inform the model, we concatenate a camera embedding to the blended feature vector, to condition the decoding also on the viewpoint information. Our experiments show that these novel model for encoding the radiance considerably improves novel view synthesis for low overlap views that are distant from the training views. Finally, we also show the capacity and convenience of our feature vector representation, demonstrating its capability not only to generate RGB values for novel views, but also their per-pixel semantic labels. Code available at https://github.com/tberriel/FeatSplat . Keywords: Gaussian Splatting, Novel View Synthesis, Feature Splatting
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20111v1">Robust Gaussian Splatting SLAM by Leveraging Loop Closure</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting algorithms excel in novel view rendering applications and have been adapted to extend the capabilities of traditional SLAM systems. However, current Gaussian Splatting SLAM methods, designed mainly for hand-held RGB or RGB-D sensors, struggle with tracking drifts when used with rotating RGB-D camera setups. In this paper, we propose a robust Gaussian Splatting SLAM architecture that utilizes inputs from rotating multiple RGB-D cameras to achieve accurate localization and photorealistic rendering performance. The carefully designed Gaussian Splatting Loop Closure module effectively addresses the issue of accumulated tracking and mapping errors found in conventional Gaussian Splatting SLAM systems. First, each Gaussian is associated with an anchor frame and categorized as historical or novel based on its timestamp. By rendering different types of Gaussians at the same viewpoint, the proposed loop detection strategy considers both co-visibility relationships and distinct rendering outcomes. Furthermore, a loop closure optimization approach is proposed to remove camera pose drift and maintain the high quality of 3D Gaussian models. The approach uses a lightweight pose graph optimization algorithm to correct pose drift and updates Gaussians based on the optimized poses. Additionally, a bundle adjustment scheme further refines camera poses using photometric and geometric constraints, ultimately enhancing the global consistency of scenarios. Quantitative and qualitative evaluations on both synthetic and real-world datasets demonstrate that our method outperforms state-of-the-art methods in camera pose estimation and novel view rendering tasks. The code will be open-sourced for the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07441v2">Instant Facial Gaussians Translator for Relightable and Interactable Facial Rendering</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 Project Page: https://dafei-qin.github.io/TransGS.github.io/
    </div>
    <details class="paper-abstract">
      We propose GauFace, a novel Gaussian Splatting representation, tailored for efficient animation and rendering of physically-based facial assets. Leveraging strong geometric priors and constrained optimization, GauFace ensures a neat and structured Gaussian representation, delivering high fidelity and real-time facial interaction of 30fps@1440p on a Snapdragon 8 Gen 2 mobile platform. Then, we introduce TransGS, a diffusion transformer that instantly translates physically-based facial assets into the corresponding GauFace representations. Specifically, we adopt a patch-based pipeline to handle the vast number of Gaussians effectively. We also introduce a novel pixel-aligned sampling scheme with UV positional encoding to ensure the throughput and rendering quality of GauFace assets generated by our TransGS. Once trained, TransGS can instantly translate facial assets with lighting conditions to GauFace representation, With the rich conditioning modalities, it also enables editing and animation capabilities reminiscent of traditional CG pipelines. We conduct extensive evaluations and user studies, compared to traditional offline and online renderers, as well as recent neural rendering methods, which demonstrate the superior performance of our approach for facial asset rendering. We also showcase diverse immersive applications of facial assets using our TransGS approach and GauFace representation, across various platforms like PCs, phones and even VR headsets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11540v4">DeRainGS: Gaussian Splatting for Enhanced Scene Reconstruction in Rainy Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-30
    </div>
    <details class="paper-abstract">
      Reconstruction under adverse rainy conditions poses significant challenges due to reduced visibility and the distortion of visual perception. These conditions can severely impair the quality of geometric maps, which is essential for applications ranging from autonomous planning to environmental monitoring. In response to these challenges, this study introduces the novel task of 3D Reconstruction in Rainy Environments (3DRRE), specifically designed to address the complexities of reconstructing 3D scenes under rainy conditions. To benchmark this task, we construct the HydroViews dataset that comprises a diverse collection of both synthesized and real-world scene images characterized by various intensities of rain streaks and raindrops. Furthermore, we propose DeRainGS, the first 3DGS method tailored for reconstruction in adverse rainy environments. Extensive experiments across a wide range of rain scenarios demonstrate that our method delivers state-of-the-art performance, remarkably outperforming existing occlusion-free methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2405.04378v4">Splat-MOVER: Multi-Stage, Open-Vocabulary Robotic Manipulation via Editable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-30
      | 💬 https://splatmover.github.io
    </div>
    <details class="paper-abstract">
      We present Splat-MOVER, a modular robotics stack for open-vocabulary robotic manipulation, which leverages the editability of Gaussian Splatting (GSplat) scene representations to enable multi-stage manipulation tasks. Splat-MOVER consists of: (i) ASK-Splat, a GSplat representation that distills semantic and grasp affordance features into the 3D scene. ASK-Splat enables geometric, semantic, and affordance understanding of 3D scenes, which is critical in many robotics tasks; (ii) SEE-Splat, a real-time scene-editing module using 3D semantic masking and infilling to visualize the motions of objects that result from robot interactions in the real-world. SEE-Splat creates a "digital twin" of the evolving environment throughout the manipulation task; and (iii) Grasp-Splat, a grasp generation module that uses ASK-Splat and SEE-Splat to propose affordance-aligned candidate grasps for open-world objects. ASK-Splat is trained in real-time from RGB images in a brief scanning phase prior to operation, while SEE-Splat and Grasp-Splat run in real-time during operation. We demonstrate the superior performance of Splat-MOVER in hardware experiments on a Kinova robot compared to two recent baselines in four single-stage, open-vocabulary manipulation tasks and in four multi-stage manipulation tasks, using the edited scene to reflect changes due to prior manipulation stages, which is not possible with existing baselines. Video demonstrations and the code for the project are available at https://splatmover.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12306v2">Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for Unconstrained Photo Collections</a></div>
    <div class="paper-meta">
      📅 2024-09-29
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Novel view synthesis from unconstrained in-the-wild image collections remains a significant yet challenging task due to photometric variations and transient occluders that complicate accurate scene reconstruction. Previous methods have approached these issues by integrating per-image appearance features embeddings in Neural Radiance Fields (NeRFs). Although 3D Gaussian Splatting (3DGS) offers faster training and real-time rendering, adapting it for unconstrained image collections is non-trivial due to the substantially different architecture. In this paper, we introduce Splatfacto-W, an approach that integrates per-Gaussian neural color features and per-image appearance embeddings into the rasterization process, along with a spherical harmonics-based background model to represent varying photometric appearances and better depict backgrounds. Our key contributions include latent appearance modeling, efficient transient object handling, and precise background modeling. Splatfacto-W delivers high-quality, real-time novel view synthesis with improved scene consistency in in-the-wild scenarios. Our method improves the Peak Signal-to-Noise Ratio (PSNR) by an average of 5.3 dB compared to 3DGS, enhances training speed by 150 times compared to NeRF-based methods, and achieves a similar rendering speed to 3DGS. Additional video results and code integrated into Nerfstudio are available at https://kevinxu02.github.io/splatfactow/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19228v1">GS-EVT: Cross-Modal Event Camera Tracking based on Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-28
    </div>
    <details class="paper-abstract">
      Reliable self-localization is a foundational skill for many intelligent mobile platforms. This paper explores the use of event cameras for motion tracking thereby providing a solution with inherent robustness under difficult dynamics and illumination. In order to circumvent the challenge of event camera-based mapping, the solution is framed in a cross-modal way. It tracks a map representation that comes directly from frame-based cameras. Specifically, the proposed method operates on top of gaussian splatting, a state-of-the-art representation that permits highly efficient and realistic novel view synthesis. The key of our approach consists of a novel pose parametrization that uses a reference pose plus first order dynamics for local differential image rendering. The latter is then compared against images of integrated events in a staggered coarse-to-fine optimization scheme. As demonstrated by our results, the realistic view rendering ability of gaussian splatting leads to stable and accurate tracking across a variety of both publicly available and newly recorded data sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18852v1">Space-time 2D Gaussian Splatting for Accurate Surface Reconstruction under Complex Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2024-09-27
      | 💬 Project page: https://tb2-sy.github.io/st-2dgs/
    </div>
    <details class="paper-abstract">
      Previous surface reconstruction methods either suffer from low geometric accuracy or lengthy training times when dealing with real-world complex dynamic scenes involving multi-person activities, and human-object interactions. To tackle the dynamic contents and the occlusions in complex scenes, we present a space-time 2D Gaussian Splatting approach. Specifically, to improve geometric quality in dynamic scenes, we learn canonical 2D Gaussian splats and deform these 2D Gaussian splats while enforcing the disks of the Gaussian located on the surface of the objects by introducing depth and normal regularizers. Further, to tackle the occlusion issues in complex scenes, we introduce a compositional opacity deformation strategy, which further reduces the surface recovery of those occluded areas. Experiments on real-world sparse-view video datasets and monocular dynamic datasets demonstrate that our reconstructions outperform state-of-the-art methods, especially for the surface of the details. The project page and more visualizations can be found at: https://tb2-sy.github.io/st-2dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19039v1">Gaussian Heritage: 3D Digitization of Cultural Heritage with Integrated Object Segmentation</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      The creation of digital replicas of physical objects has valuable applications for the preservation and dissemination of tangible cultural heritage. However, existing methods are often slow, expensive, and require expert knowledge. We propose a pipeline to generate a 3D replica of a scene using only RGB images (e.g. photos of a museum) and then extract a model for each item of interest (e.g. pieces in the exhibit). We do this by leveraging the advancements in novel view synthesis and Gaussian Splatting, modified to enable efficient 3D segmentation. This approach does not need manual annotation, and the visual inputs can be captured using a standard smartphone, making it both affordable and easy to deploy. We provide an overview of the method and baseline evaluation of the accuracy of object segmentation. The code is available at https://mahtaabdn.github.io/gaussian_heritage.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02370v4">Query-based Semantic Gaussian Field for Scene Representation in Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Latent scene representation plays a significant role in training reinforcement learning (RL) agents. To obtain good latent vectors describing the scenes, recent works incorporate the 3D-aware latent-conditioned NeRF pipeline into scene representation learning. However, these NeRF-related methods struggle to perceive 3D structural information due to the inefficient dense sampling in volumetric rendering. Moreover, they lack fine-grained semantic information included in their scene representation vectors because they evenly consider free and occupied spaces. Both of them can destroy the performance of downstream RL tasks. To address the above challenges, we propose a novel framework that adopts the efficient 3D Gaussian Splatting (3DGS) to learn 3D scene representation for the first time. In brief, we present the Query-based Generalizable 3DGS to bridge the 3DGS technique and scene representations with more geometrical awareness than those in NeRFs. Moreover, we present the Hierarchical Semantics Encoding to ground the fine-grained semantic features to 3D Gaussians and further distilled to the scene representation vectors. We conduct extensive experiments on two RL platforms including Maniskill2 and Robomimic across 10 different tasks. The results show that our method outperforms the other 5 baselines by a large margin. We achieve the best success rates on 8 tasks and the second-best on the other two tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11247v2">Compact 3D Gaussian Splatting For Dense Visual SLAM</a></div>
    <div class="paper-meta">
      📅 2024-09-27
    </div>
    <details class="paper-abstract">
      Recent work has shown that 3D Gaussian-based SLAM enables high-quality reconstruction, accurate pose estimation, and real-time rendering of scenes. However, these approaches are built on a tremendous number of redundant 3D Gaussian ellipsoids, leading to high memory and storage costs, and slow training speed. To address the limitation, we propose a compact 3D Gaussian Splatting SLAM system that reduces the number and the parameter size of Gaussian ellipsoids. A sliding window-based masking strategy is first proposed to reduce the redundant ellipsoids. Then we observe that the covariance matrix (geometry) of most 3D Gaussian ellipsoids are extremely similar, which motivates a novel geometry codebook to compress 3D Gaussian geometric attributes, i.e., the parameters. Robust and accurate pose estimation is achieved by a global bundle adjustment method with reprojection loss. Extensive experiments demonstrate that our method achieves faster training and rendering speed while maintaining the state-of-the-art (SOTA) quality of the scene representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04378v4">Splat-MOVER: Multi-Stage, Open-Vocabulary Robotic Manipulation via Editable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 https://splatmover.github.io
    </div>
    <details class="paper-abstract">
      We present Splat-MOVER, a modular robotics stack for open-vocabulary robotic manipulation, which leverages the editability of Gaussian Splatting (GSplat) scene representations to enable multi-stage manipulation tasks. Splat-MOVER consists of: (i) ASK-Splat, a GSplat representation that distills semantic and grasp affordance features into the 3D scene. ASK-Splat enables geometric, semantic, and affordance understanding of 3D scenes, which is critical in many robotics tasks; (ii) SEE-Splat, a real-time scene-editing module using 3D semantic masking and infilling to visualize the motions of objects that result from robot interactions in the real-world. SEE-Splat creates a "digital twin" of the evolving environment throughout the manipulation task; and (iii) Grasp-Splat, a grasp generation module that uses ASK-Splat and SEE-Splat to propose affordance-aligned candidate grasps for open-world objects. ASK-Splat is trained in real-time from RGB images in a brief scanning phase prior to operation, while SEE-Splat and Grasp-Splat run in real-time during operation. We demonstrate the superior performance of Splat-MOVER in hardware experiments on a Kinova robot compared to two recent baselines in four single-stage, open-vocabulary manipulation tasks and in four multi-stage manipulation tasks, using the edited scene to reflect changes due to prior manipulation stages, which is not possible with existing baselines. Video demonstrations and the code for the project are available at https://splatmover.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18159v3">CompGS: Smaller and Faster Gaussian Splatting with Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 Code is available at https://github.com/UCDvision/compact3d
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a new method for modeling and rendering 3D radiance fields that achieves much faster learning and rendering time compared to SOTA NeRF methods. However, it comes with a drawback in the much larger storage demand compared to NeRF methods since it needs to store the parameters for several 3D Gaussians. We notice that many Gaussians may share similar parameters, so we introduce a simple vector quantization method based on K-means to quantize the Gaussian parameters while optimizing them. Then, we store the small codebook along with the index of the code for each Gaussian. We compress the indices further by sorting them and using a method similar to run-length encoding. Moreover, we use a simple regularizer to encourage zero opacity (invisible Gaussians) to reduce the storage and rendering time by a large factor through reducing the number of Gaussians. We do extensive experiments on standard benchmarks as well as an existing 3D dataset that is an order of magnitude larger than the standard benchmarks used in this field. We show that our simple yet effective method can reduce the storage cost for 3DGS by 40 to 50x and rendering time by 2 to 3x with a very small drop in the quality of rendered images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18122v1">RT-GuIDE: Real-Time Gaussian splatting for Information-Driven Exploration</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 Submitted to ICRA2025
    </div>
    <details class="paper-abstract">
      We propose a framework for active mapping and exploration that leverages Gaussian splatting for constructing information-rich maps. Further, we develop a parallelized motion planning algorithm that can exploit the Gaussian map for real-time navigation. The Gaussian map constructed onboard the robot is optimized for both photometric and geometric quality while enabling real-time situational awareness for autonomy. We show through simulation experiments that our method is competitive with approaches that use alternate information gain metrics, while being orders of magnitude faster to compute. In real-world experiments, our algorithm achieves better map quality (10% higher Peak Signal-to-Noise Ratio (PSNR) and 30% higher geometric reconstruction accuracy) than Gaussian maps constructed by traditional exploration baselines. Experiment videos and more details can be found on our project page: https://tyuezhan.github.io/RT_GuIDE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18108v1">Language-Embedded Gaussian Splats (LEGS): Incrementally Building Room-Scale Representations with a Mobile Robot</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      Building semantic 3D maps is valuable for searching for objects of interest in offices, warehouses, stores, and homes. We present a mapping system that incrementally builds a Language-Embedded Gaussian Splat (LEGS): a detailed 3D scene representation that encodes both appearance and semantics in a unified representation. LEGS is trained online as a robot traverses its environment to enable localization of open-vocabulary object queries. We evaluate LEGS on 4 room-scale scenes where we query for objects in the scene to assess how LEGS can capture semantic meaning. We compare LEGS to LERF and find that while both systems have comparable object query success rates, LEGS trains over 3.5x faster than LERF. Results suggest that a multi-camera setup and incremental bundle adjustment can boost visual reconstruction quality in constrained robot trajectories, and suggest LEGS can localize open-vocabulary and long-tail object queries with up to 66% accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17917v1">WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      While style transfer techniques have been well-developed for 2D image stylization, the extension of these methods to 3D scenes remains relatively unexplored. Existing approaches demonstrate proficiency in transferring colors and textures but often struggle with replicating the geometry of the scenes. In our work, we leverage an explicit Gaussian Splatting (GS) representation and directly match the distributions of Gaussians between style and content scenes using the Earth Mover's Distance (EMD). By employing the entropy-regularized Wasserstein-2 distance, we ensure that the transformation maintains spatial smoothness. Additionally, we decompose the scene stylization problem into smaller chunks to enhance efficiency. This paradigm shift reframes stylization from a pure generative process driven by latent space losses to an explicit matching of distributions between two Gaussian representations. Our method achieves high-resolution 3D stylization by faithfully transferring details from 3D style scenes onto the content scene. Furthermore, WaSt-3D consistently delivers results across diverse content and style scenes without necessitating any training, as it relies solely on optimization-based techniques. See our project page for additional results and source code: $\href{https://compvis.github.io/wast3d/}{https://compvis.github.io/wast3d/}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.04564v3">EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS</a></div>
    <div class="paper-meta">
      📅 2024-09-26
      | 💬 Website: https://efficientgaussian.github.io Code: https://github.com/Sharath-girish/efficientgaussian
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting (3D-GS) has gained popularity in novel-view scene synthesis. It addresses the challenges of lengthy training times and slow rendering speeds associated with Neural Radiance Fields (NeRFs). Through rapid, differentiable rasterization of 3D Gaussians, 3D-GS achieves real-time rendering and accelerated training. They, however, demand substantial memory resources for both training and storage, as they require millions of Gaussians in their point cloud representation for each scene. We present a technique utilizing quantized embeddings to significantly reduce per-point memory storage requirements and a coarse-to-fine training strategy for a faster and more stable optimization of the Gaussian point clouds. Our approach develops a pruning stage which results in scene representations with fewer Gaussians, leading to faster training times and rendering speeds for real-time rendering of high resolution scenes. We reduce storage memory by more than an order of magnitude all while preserving the reconstruction quality. We validate the effectiveness of our approach on a variety of datasets and scenes preserving the visual quality while consuming 10-20x lesser memory and faster training/inference speed. Project page and code is available https://efficientgaussian.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06926v2">Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      In this paper, we present a real-time photo-realistic SLAM method based on marrying Gaussian Splatting with LiDAR-Inertial-Camera SLAM. Most existing radiance-field-based SLAM systems mainly focus on bounded indoor environments, equipped with RGB-D or RGB sensors. However, they are prone to decline when expanding to unbounded scenes or encountering adverse conditions, such as violent motions and changing illumination. In contrast, oriented to general scenarios, our approach additionally tightly fuses LiDAR, IMU, and camera for robust pose estimation and photo-realistic online mapping. To compensate for regions unobserved by the LiDAR, we propose to integrate both the triangulated visual points from images and LiDAR points for initializing 3D Gaussians. In addition, the modeling of the sky and varying camera exposure have been realized for high-quality rendering. Notably, we implement our system purely with C++ and CUDA, and meticulously design a series of strategies to accelerate the online optimization of the Gaussian-based scene representation. Extensive experiments demonstrate that our method outperforms its counterparts while maintaining real-time capability. Impressively, regarding photo-realistic mapping, our method with our estimated poses even surpasses all the compared approaches that utilize privileged ground-truth poses for mapping. Our code will be released on project page https://xingxingzuo.github.io/gaussian_lic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06945v2">Direct Learning of Mesh and Appearance via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      Accurately reconstructing a 3D scene including explicit geometry information is both attractive and challenging. Geometry reconstruction can benefit from incorporating differentiable appearance models, such as Neural Radiance Fields and 3D Gaussian Splatting (3DGS). However, existing methods encounter efficiency issues due to indirect geometry learning and the paradigm of separately modeling geometry and surface appearance. In this work, we propose a learnable scene model that incorporates 3DGS with an explicit geometry representation, namely a mesh. Our model learns the mesh and appearance in an end-to-end manner, where we bind 3D Gaussians to the mesh faces and perform differentiable rendering of 3DGS to obtain photometric supervision. The model creates an effective information pathway to supervise the learning of both 3DGS and mesh. Experimental results demonstrate that the learned scene model not only achieves state-of-the-art efficiency and rendering quality but also supports manipulation using the explicit mesh. In addition, our model has a unique advantage in adapting to scene updates, thanks to the end-to-end learning of both mesh and appearance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15435v2">Enhancement of 3D Gaussian Splatting using Raw Mesh for Photorealistic Recreation of Architectures</a></div>
    <div class="paper-meta">
      📅 2024-09-26
    </div>
    <details class="paper-abstract">
      The photorealistic reconstruction and rendering of architectural scenes have extensive applications in industries such as film, games, and transportation. It also plays an important role in urban planning, architectural design, and the city's promotion, especially in protecting historical and cultural relics. The 3D Gaussian Splatting, due to better performance over NeRF, has become a mainstream technology in 3D reconstruction. Its only input is a set of images but it relies heavily on geometric parameters computed by the SfM process. At the same time, there is an existing abundance of raw 3D models, that could inform the structural perception of certain buildings but cannot be applied. In this paper, we propose a straightforward method to harness these raw 3D models to guide 3D Gaussians in capturing the basic shape of the building and improve the visual quality of textures and details when photos are captured non-systematically. This exploration opens up new possibilities for improving the effectiveness of 3D reconstruction techniques in the field of architectural design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17345v1">SeaSplat: Representing Underwater Scenes with 3D Gaussian Splatting and a Physically Grounded Image Formation Model</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Project page here: https://seasplat.github.io
    </div>
    <details class="paper-abstract">
      We introduce SeaSplat, a method to enable real-time rendering of underwater scenes leveraging recent advances in 3D radiance fields. Underwater scenes are challenging visual environments, as rendering through a medium such as water introduces both range and color dependent effects on image capture. We constrain 3D Gaussian Splatting (3DGS), a recent advance in radiance fields enabling rapid training and real-time rendering of full 3D scenes, with a physically grounded underwater image formation model. Applying SeaSplat to the real-world scenes from SeaThru-NeRF dataset, a scene collected by an underwater vehicle in the US Virgin Islands, and simulation-degraded real-world scenes, not only do we see increased quantitative performance on rendering novel viewpoints from the scene with the medium present, but are also able to recover the underlying true color of the scene and restore renders to be without the presence of the intervening medium. We show that the underwater image formation helps learn scene structure, with better depth maps, as well as show that our improvements maintain the significant computational improvements afforded by leveraging a 3D Gaussian representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17280v1">Disco4D: Disentangled 4D Human Generation and Animation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      We present \textbf{Disco4D}, a novel Gaussian Splatting framework for 4D human generation and animation from a single image. Different from existing methods, Disco4D distinctively disentangles clothings (with Gaussian models) from the human body (with SMPL-X model), significantly enhancing the generation details and flexibility. It has the following technical innovations. \textbf{1)} Disco4D learns to efficiently fit the clothing Gaussians over the SMPL-X Gaussians. \textbf{2)} It adopts diffusion models to enhance the 3D generation process, \textit{e.g.}, modeling occluded parts not visible in the input image. \textbf{3)} It learns an identity encoding for each clothing Gaussian to facilitate the separation and extraction of clothing assets. Furthermore, Disco4D naturally supports 4D human animation with vivid dynamics. Extensive experiments demonstrate the superiority of Disco4D on 4D human generation and animation tasks. Our visualizations can be found in \url{https://disco-4d.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16944v1">Go-SLAM: Grounded Object Segmentation and Localization with Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2024-09-25
    </div>
    <details class="paper-abstract">
      We introduce Go-SLAM, a novel framework that utilizes 3D Gaussian Splatting SLAM to reconstruct dynamic environments while embedding object-level information within the scene representations. This framework employs advanced object segmentation techniques, assigning a unique identifier to each Gaussian splat that corresponds to the object it represents. Consequently, our system facilitates open-vocabulary querying, allowing users to locate objects using natural language descriptions. Furthermore, the framework features an optimal path generation module that calculates efficient navigation paths for robots toward queried objects, considering obstacles and environmental uncertainties. Comprehensive evaluations in various scene settings demonstrate the effectiveness of our approach in delivering high-fidelity scene reconstructions, precise object segmentation, flexible object querying, and efficient robot path planning. This work represents an additional step forward in bridging the gap between 3D scene reconstruction, semantic object understanding, and real-time environment interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16938v1">Generative Object Insertion in Gaussian Splatting with a Multi-View Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Project Page: https://github.com/JiuTongBro/MultiView_Inpaint
    </div>
    <details class="paper-abstract">
      Generating and inserting new objects into 3D content is a compelling approach for achieving versatile scene recreation. Existing methods, which rely on SDS optimization or single-view inpainting, often struggle to produce high-quality results. To address this, we propose a novel method for object insertion in 3D content represented by Gaussian Splatting. Our approach introduces a multi-view diffusion model, dubbed MVInpainter, which is built upon a pre-trained stable video diffusion model to facilitate view-consistent object inpainting. Within MVInpainter, we incorporate a ControlNet-based conditional injection module to enable controlled and more predictable multi-view generation. After generating the multi-view inpainted results, we further propose a mask-aware 3D reconstruction technique to refine Gaussian Splatting reconstruction from these sparse inpainted views. By leveraging these fabricate techniques, our approach yields diverse results, ensures view-consistent and harmonious insertions, and produces better object quality. Extensive experiments demonstrate that our approach outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16915v1">Let's Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 First two authors contributed equally. Project Page: https://roahmlab.github.io/splanning
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields and Gaussian Splatting have transformed the field of computer vision by enabling photo-realistic representation of complex scenes. Despite this success, they have seen only limited use in real-world robotics tasks such as trajectory optimization. Two key factors have contributed to this limited success. First, it is challenging to reason about collisions in radiance models. Second, it is difficult to perform inference of radiance models fast enough for real-time trajectory synthesis. This paper addresses these challenges by proposing SPLANNING, a risk-aware trajectory optimizer that operates in a Gaussian Splatting model. This paper first derives a method for rigorously upper-bounding the probability of collision between a robot and a radiance field. Second, this paper introduces a normalized reformulation of Gaussian Splatting that enables the efficient computation of the collision bound in a Gaussian Splat. Third, a method is presented to optimize trajectories while avoiding collisions with a scene represented by a Gaussian Splat. Experiments demonstrate that SPLANNING outperforms state-of-the-art methods in generating collision-free trajectories in highly cluttered environments. The proposed system is also tested on a real-world robot manipulator. A project page is available at https://roahmlab.github.io/splanning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13827v2">Identifying Unnecessary 3D Gaussians using Clustering for Fast Rendering of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 Our claim that Step 1 of 3D Gaussian splatting accounts for ~50% of rendering (Fig. 2) was incorrect. Rerunning simulations showed it's only ~20%. Consequently, our method's performance decreased by ~40% from initial reports. We're exploring new directions but have no concrete plans yet. To avoid reader confusion, we're withdrawing the paper and will resubmit once revised
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3D-GS) is a new rendering approach that outperforms the neural radiance field (NeRF) in terms of both speed and image quality. 3D-GS represents 3D scenes by utilizing millions of 3D Gaussians and projects these Gaussians onto the 2D image plane for rendering. However, during the rendering process, a substantial number of unnecessary 3D Gaussians exist for the current view direction, resulting in significant computation costs associated with their identification. In this paper, we propose a computational reduction technique that quickly identifies unnecessary 3D Gaussians in real-time for rendering the current view without compromising image quality. This is accomplished through the offline clustering of 3D Gaussians that are close in distance, followed by the projection of these clusters onto a 2D image plane during runtime. Additionally, we analyze the bottleneck associated with the proposed technique when executed on GPUs and propose an efficient hardware architecture that seamlessly supports the proposed scheme. For the Mip-NeRF360 dataset, the proposed technique excludes 63% of 3D Gaussians on average before the 2D image projection, which reduces the overall rendering computation by almost 38.3% without sacrificing peak-signal-to-noise-ratio (PSNR). The proposed accelerator also achieves a speedup of 10.7x compared to a GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2401.00834v3">Deblurring 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-25
      | 💬 29 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Recent studies in Radiance Fields have paved the robust way for novel view synthesis with their photorealistic rendering quality. Nevertheless, they usually employ neural networks and volumetric rendering, which are costly to train and impede their broad use in various real-time applications due to the lengthy rendering time. Lately 3D Gaussians splatting-based approach has been proposed to model the 3D scene, and it achieves remarkable visual quality while rendering the images in real-time. However, it suffers from severe degradation in the rendering quality if the training images are blurry. Blurriness commonly occurs due to the lens defocusing, object motion, and camera shake, and it inevitably intervenes in clean image acquisition. Several previous studies have attempted to render clean and sharp images from blurry input images using neural fields. The majority of those works, however, are designed only for volumetric rendering-based neural radiance fields and are not straightforwardly applicable to rasterization-based 3D Gaussian splatting methods. Thus, we propose a novel real-time deblurring framework, Deblurring 3D Gaussian Splatting, using a small Multi-Layer Perceptron (MLP) that manipulates the covariance of each 3D Gaussian to model the scene blurriness. While Deblurring 3D Gaussian Splatting can still enjoy real-time rendering, it can reconstruct fine and sharp details from blurry images. A variety of experiments have been conducted on the benchmark, and the results have revealed the effectiveness of our approach for deblurring. Qualitative results are available at https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16502v1">GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Project website at https://gsplatloc.github.io/
    </div>
    <details class="paper-abstract">
      Although various visual localization approaches exist, such as scene coordinate and pose regression, these methods often struggle with high memory consumption or extensive optimization requirements. To address these challenges, we utilize recent advancements in novel view synthesis, particularly 3D Gaussian Splatting (3DGS), to enhance localization. 3DGS allows for the compact encoding of both 3D geometry and scene appearance with its spatial features. Our method leverages the dense description maps produced by XFeat's lightweight keypoint detection and description model. We propose distilling these dense keypoint descriptors into 3DGS to improve the model's spatial understanding, leading to more accurate camera pose predictions through 2D-3D correspondences. After estimating an initial pose, we refine it using a photometric warping loss. Benchmarking on popular indoor and outdoor datasets shows that our approach surpasses state-of-the-art Neural Render Pose (NRP) methods, including NeRFMatch and PNeRFLoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16470v1">Frequency-based View Selection in Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Three-dimensional reconstruction is a fundamental problem in robotics perception. We examine the problem of active view selection to perform 3D Gaussian Splatting reconstructions with as few input images as possible. Although 3D Gaussian Splatting has made significant progress in image rendering and 3D reconstruction, the quality of the reconstruction is strongly impacted by the selection of 2D images and the estimation of camera poses through Structure-from-Motion (SfM) algorithms. Current methods to select views that rely on uncertainties from occlusions, depth ambiguities, or neural network predictions directly are insufficient to handle the issue and struggle to generalize to new scenes. By ranking the potential views in the frequency domain, we are able to effectively estimate the potential information gain of new viewpoints without ground truth data. By overcoming current constraints on model architecture and efficacy, our method achieves state-of-the-art results in view selection, demonstrating its potential for efficient image-based 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12774v3">GaRField++: Reinforced Gaussian Radiance Fields for Large-Scale 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      This paper proposes a novel framework for large-scale scene reconstruction based on 3D Gaussian splatting (3DGS) and aims to address the scalability and accuracy challenges faced by existing methods. For tackling the scalability issue, we split the large scene into multiple cells, and the candidate point-cloud and camera views of each cell are correlated through a visibility-based camera selection and a progressive point-cloud extension. To reinforce the rendering quality, three highlighted improvements are made in comparison with vanilla 3DGS, which are a strategy of the ray-Gaussian intersection and the novel Gaussians density control for learning efficiency, an appearance decoupling module based on ConvKAN network to solve uneven lighting conditions in large-scale scenes, and a refined final loss with the color loss, the depth distortion loss, and the normal consistency loss. Finally, the seamless stitching procedure is executed to merge the individual Gaussian radiance field for novel view synthesis across different cells. Evaluation of Mill19, Urban3D, and MatrixCity datasets shows that our method consistently generates more high-fidelity rendering results than state-of-the-art methods of large-scale scene reconstruction. We further validate the generalizability of the proposed approach by rendering on self-collected video clips recorded by a commercial drone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09377v2">GGHead: Fast and Generalizable 3D Gaussian Heads</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Project Page: https://tobias-kirschstein.github.io/gghead/ ; YouTube Video: https://youtu.be/M5vq3DoZ7RI
    </div>
    <details class="paper-abstract">
      Learning 3D head priors from large 2D image collections is an important step towards high-quality 3D-aware human modeling. A core requirement is an efficient architecture that scales well to large-scale datasets and large image resolutions. Unfortunately, existing 3D GANs struggle to scale to generate samples at high resolutions due to their relatively slow train and render speeds, and typically have to rely on 2D superresolution networks at the expense of global 3D consistency. To address these challenges, we propose Generative Gaussian Heads (GGHead), which adopts the recent 3D Gaussian Splatting representation within a 3D GAN framework. To generate a 3D representation, we employ a powerful 2D CNN generator to predict Gaussian attributes in the UV space of a template head mesh. This way, GGHead exploits the regularity of the template's UV layout, substantially facilitating the challenging task of predicting an unstructured set of 3D Gaussians. We further improve the geometric fidelity of the generated 3D representations with a novel total variation loss on rendered UV coordinates. Intuitively, this regularization encourages that neighboring rendered pixels should stem from neighboring Gaussians in the template's UV space. Taken together, our pipeline can efficiently generate 3D heads trained only from single-view 2D image observations. Our proposed framework matches the quality of existing 3D head GANs on FFHQ while being both substantially faster and fully 3D consistent. As a result, we demonstrate real-time generation and rendering of high-quality 3D-consistent heads at $1024^2$ resolution for the first time. Project Website: https://tobias-kirschstein.github.io/gghead
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15959v1">Semantics-Controlled Gaussian Splatting for Outdoor Scene Reconstruction and Rendering in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Advancements in 3D rendering like Gaussian Splatting (GS) allow novel view synthesis and real-time rendering in virtual reality (VR). However, GS-created 3D environments are often difficult to edit. For scene enhancement or to incorporate 3D assets, segmenting Gaussians by class is essential. Existing segmentation approaches are typically limited to certain types of scenes, e.g., ''circular'' scenes, to determine clear object boundaries. However, this method is ineffective when removing large objects in non-''circling'' scenes such as large outdoor scenes. We propose Semantics-Controlled GS (SCGS), a segmentation-driven GS approach, enabling the separation of large scene parts in uncontrolled, natural environments. SCGS allows scene editing and the extraction of scene parts for VR. Additionally, we introduce a challenging outdoor dataset, overcoming the ''circling'' setup. We outperform the state-of-the-art in visual quality on our dataset and in segmentation quality on the 3D-OVS dataset. We conducted an exploratory user study, comparing a 360-video, plain GS, and SCGS in VR with a fixed viewpoint. In our subsequent main study, users were allowed to move freely, evaluating plain GS and SCGS. Our main study results show that participants clearly prefer SCGS over plain GS. We overall present an innovative approach that surpasses the state-of-the-art both technically and in user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.00834v3">Deblurring 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 29 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Recent studies in Radiance Fields have paved the robust way for novel view synthesis with their photorealistic rendering quality. Nevertheless, they usually employ neural networks and volumetric rendering, which are costly to train and impede their broad use in various real-time applications due to the lengthy rendering time. Lately 3D Gaussians splatting-based approach has been proposed to model the 3D scene, and it achieves remarkable visual quality while rendering the images in real-time. However, it suffers from severe degradation in the rendering quality if the training images are blurry. Blurriness commonly occurs due to the lens defocusing, object motion, and camera shake, and it inevitably intervenes in clean image acquisition. Several previous studies have attempted to render clean and sharp images from blurry input images using neural fields. The majority of those works, however, are designed only for volumetric rendering-based neural radiance fields and are not straightforwardly applicable to rasterization-based 3D Gaussian splatting methods. Thus, we propose a novel real-time deblurring framework, Deblurring 3D Gaussian Splatting, using a small Multi-Layer Perceptron (MLP) that manipulates the covariance of each 3D Gaussian to model the scene blurriness. While Deblurring 3D Gaussian Splatting can still enjoy real-time rendering, it can reconstruct fine and sharp details from blurry images. A variety of experiments have been conducted on the benchmark, and the results have revealed the effectiveness of our approach for deblurring. Qualitative results are available at https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15689v1">Plenoptic PNG: Real-Time Neural Radiance Fields in 150 KB</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      The goal of this paper is to encode a 3D scene into an extremely compact representation from 2D images and to enable its transmittance, decoding and rendering in real-time across various platforms. Despite the progress in NeRFs and Gaussian Splats, their large model size and specialized renderers make it challenging to distribute free-viewpoint 3D content as easily as images. To address this, we have designed a novel 3D representation that encodes the plenoptic function into sinusoidal function indexed dense volumes. This approach facilitates feature sharing across different locations, improving compactness over traditional spatial voxels. The memory footprint of the dense 3D feature grid can be further reduced using spatial decomposition techniques. This design combines the strengths of spatial hashing functions and voxel decomposition, resulting in a model size as small as 150 KB for each 3D scene. Moreover, PPNG features a lightweight rendering pipeline with only 300 lines of code that decodes its representation into standard GL textures and fragment shaders. This enables real-time rendering using the traditional GL pipeline, ensuring universal compatibility and efficiency across various platforms without additional dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07160v1">TextToon: Real-Time Text Toonify Head Avatar from Single Video</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 Project Page: https://songluchuan.github.io/TextToon/
    </div>
    <details class="paper-abstract">
      We propose TextToon, a method to generate a drivable toonified avatar. Given a short monocular video sequence and a written instruction about the avatar style, our model can generate a high-fidelity toonified avatar that can be driven in real-time by another video with arbitrary identities. Existing related works heavily rely on multi-view modeling to recover geometry via texture embeddings, presented in a static manner, leading to control limitations. The multi-view video input also makes it difficult to deploy these models in real-world applications. To address these issues, we adopt a conditional embedding Tri-plane to learn realistic and stylized facial representations in a Gaussian deformation field. Additionally, we expand the stylization capabilities of 3D Gaussian Splatting by introducing an adaptive pixel-translation neural network and leveraging patch-aware contrastive learning to achieve high-quality images. To push our work into consumer applications, we develop a real-time system that can operate at 48 FPS on a GPU machine and 15-18 FPS on a mobile machine. Extensive experiments demonstrate the efficacy of our approach in generating textual avatars over existing methods in terms of quality and real-time animation. Please refer to our project page for more details: https://songluchuan.github.io/TextToon/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14778v1">Human Hair Reconstruction with Strand-Aligned 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      We introduce a new hair modeling method that uses a dual representation of classical hair strands and 3D Gaussians to produce accurate and realistic strand-based reconstructions from multi-view data. In contrast to recent approaches that leverage unstructured Gaussians to model human avatars, our method reconstructs the hair using 3D polylines, or strands. This fundamental difference allows the use of the resulting hairstyles out-of-the-box in modern computer graphics engines for editing, rendering, and simulation. Our 3D lifting method relies on unstructured Gaussians to generate multi-view ground truth data to supervise the fitting of hair strands. The hairstyle itself is represented in the form of the so-called strand-aligned 3D Gaussians. This representation allows us to combine strand-based hair priors, which are essential for realistic modeling of the inner structure of hairstyles, with the differentiable rendering capabilities of 3D Gaussian Splatting. Our method, named Gaussian Haircut, is evaluated on synthetic and real scenes and demonstrates state-of-the-art performance in the task of strand-based hair reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00381v3">3D Gaussian Splatting for Large-scale Surface Reconstruction from Aerial Images</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has demonstrated excellent ability in small-scale 3D surface reconstruction. However, extending 3DGS to large-scale scenes remains a significant challenge. To address this gap, we propose a novel 3DGS-based method for large-scale surface reconstruction using aerial multi-view stereo (MVS) images, named Aerial Gaussian Splatting (AGS). First, we introduce a data chunking method tailored for large-scale aerial images, making 3DGS feasible for surface reconstruction over extensive scenes. Second, we integrate the Ray-Gaussian Intersection method into 3DGS to obtain depth and normal information. Finally, we implement multi-view geometric consistency constraints to enhance the geometric consistency across different views. Our experiments on multiple datasets demonstrate, for the first time, the 3DGS-based method can match conventional aerial MVS methods on geometric accuracy in aerial large-scale surface reconstruction, and our method also beats state-of-the-art GS-based methods both on geometry and rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09733v2">Unified Gaussian Primitives for Scene Representation and Rendering</a></div>
    <div class="paper-meta">
      📅 2024-09-22
    </div>
    <details class="paper-abstract">
      Searching for a unified scene representation remains a research challenge in computer graphics. Traditional mesh-based representations are unsuitable for dense, fuzzy elements, and introduce additional complexity for filtering and differentiable rendering. Conversely, voxel-based representations struggle to model hard surfaces and suffer from intensive memory requirement. We propose a general-purpose rendering primitive based on 3D Gaussian distribution for unified scene representation, featuring versatile appearance ranging from glossy surfaces to fuzzy elements, as well as physically based scattering to enable accurate global illumination. We formulate the rendering theory for the primitive based on non-exponential transport and derive efficient rendering operations to be compatible with Monte Carlo path tracing. The new representation can be converted from different sources, including meshes and 3D Gaussian splatting, and further refined via transmittance optimization thanks to its differentiability. We demonstrate the versatility of our representation in various rendering applications such as global illumination and appearance editing, while supporting arbitrary lighting conditions by nature. Additionally, we compare our representation to existing volumetric representations, highlighting its efficiency to reproduce details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14316v1">MVPGS: Excavating Multi-view Priors for Gaussian Splatting from Sparse Input Views</a></div>
    <div class="paper-meta">
      📅 2024-09-22
      | 💬 Accepted by ECCV 2024, Project page: https://zezeaaa.github.io/projects/MVPGS/
    </div>
    <details class="paper-abstract">
      Recently, the Neural Radiance Field (NeRF) advancement has facilitated few-shot Novel View Synthesis (NVS), which is a significant challenge in 3D vision applications. Despite numerous attempts to reduce the dense input requirement in NeRF, it still suffers from time-consumed training and rendering processes. More recently, 3D Gaussian Splatting (3DGS) achieves real-time high-quality rendering with an explicit point-based representation. However, similar to NeRF, it tends to overfit the train views for lack of constraints. In this paper, we propose \textbf{MVPGS}, a few-shot NVS method that excavates the multi-view priors based on 3D Gaussian Splatting. We leverage the recent learning-based Multi-view Stereo (MVS) to enhance the quality of geometric initialization for 3DGS. To mitigate overfitting, we propose a forward-warping method for additional appearance constraints conforming to scenes based on the computed geometry. Furthermore, we introduce a view-consistent geometry constraint for Gaussian parameters to facilitate proper optimization convergence and utilize a monocular depth regularization as compensation. Experiments show that the proposed method achieves state-of-the-art performance with real-time rendering speed. Project page: https://zezeaaa.github.io/projects/MVPGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14067v1">SplatLoc: 3D Gaussian Splatting-based Visual Localization for Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2024-09-21
    </div>
    <details class="paper-abstract">
      Visual localization plays an important role in the applications of Augmented Reality (AR), which enable AR devices to obtain their 6-DoF pose in the pre-build map in order to render virtual content in real scenes. However, most existing approaches can not perform novel view rendering and require large storage capacities for maps. To overcome these limitations, we propose an efficient visual localization method capable of high-quality rendering with fewer parameters. Specifically, our approach leverages 3D Gaussian primitives as the scene representation. To ensure precise 2D-3D correspondences for pose estimation, we develop an unbiased 3D scene-specific descriptor decoder for Gaussian primitives, distilled from a constructed feature volume. Additionally, we introduce a salient 3D landmark selection algorithm that selects a suitable primitive subset based on the saliency score for localization. We further regularize key Gaussian primitives to prevent anisotropic effects, which also improves localization performance. Extensive experiments on two widely used datasets demonstrate that our method achieves superior or comparable rendering and localization performance to state-of-the-art implicit-based visual localization approaches. Project page: \href{https://zju3dv.github.io/splatloc}{https://zju3dv.github.io/splatloc}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13995v2">Avatar Concept Slider: Manipulate Concepts In Your Human Avatar With Fine-grained Control</a></div>
    <div class="paper-meta">
      📅 2024-09-21
    </div>
    <details class="paper-abstract">
      Language based editing of 3D human avatars to precisely match user requirements is challenging due to the inherent ambiguity and limited expressiveness of natural language. To overcome this, we propose the Avatar Concept Slider (ACS), a 3D avatar editing method that allows precise manipulation of semantic concepts in human avatars towards a specified intermediate point between two extremes of concepts, akin to moving a knob along a slider track. To achieve this, our ACS has three designs. 1) A Concept Sliding Loss based on Linear Discriminant Analysis to pinpoint the concept-specific axis for precise editing. 2) An Attribute Preserving Loss based on Principal Component Analysis for improved preservation of avatar identity during editing. 3) A 3D Gaussian Splatting primitive selection mechanism based on concept-sensitivity, which updates only the primitives that are the most sensitive to our target concept, to improve efficiency. Results demonstrate that our ACS enables fine-grained 3D avatar editing with efficient feedback, without harming the avatar quality or compromising the avatar's identifying attributes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13392v1">Elite-EvGS: Learning Event-based 3D Gaussian Splatting by Distilling Event-to-Video Priors</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Event cameras are bio-inspired sensors that output asynchronous and sparse event streams, instead of fixed frames. Benefiting from their distinct advantages, such as high dynamic range and high temporal resolution, event cameras have been applied to address 3D reconstruction, important for robotic mapping. Recently, neural rendering techniques, such as 3D Gaussian splatting (3DGS), have been shown successful in 3D reconstruction. However, it still remains under-explored how to develop an effective event-based 3DGS pipeline. In particular, as 3DGS typically depends on high-quality initialization and dense multiview constraints, a potential problem appears for the 3DGS optimization with events given its inherent sparse property. To this end, we propose a novel event-based 3DGS framework, named Elite-EvGS. Our key idea is to distill the prior knowledge from the off-the-shelf event-to-video (E2V) models to effectively reconstruct 3D scenes from events in a coarse-to-fine optimization manner. Specifically, to address the complexity of 3DGS initialization from events, we introduce a novel warm-up initialization strategy that optimizes a coarse 3DGS from the frames generated by E2V models and then incorporates events to refine the details. Then, we propose a progressive event supervision strategy that employs the window-slicing operation to progressively reduce the number of events used for supervision. This subtly relives the temporal randomness of the event frames, benefiting the optimization of local textural and global structural details. Experiments on the benchmark datasets demonstrate that Elite-EvGS can reconstruct 3D scenes with better textural and structural details. Meanwhile, our method yields plausible performance on the captured real-world data, including diverse challenging conditions, such as fast motion and low light scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13055v1">MGSO: Monocular Real-time Photometric SLAM with Efficient 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 Paper Contribution to the ICRA 2025 Conference. Currently being reviewed
    </div>
    <details class="paper-abstract">
      Real-time SLAM with dense 3D mapping is computationally challenging, especially on resource-limited devices. The recent development of 3D Gaussian Splatting (3DGS) offers a promising approach for real-time dense 3D reconstruction. However, existing 3DGS-based SLAM systems struggle to balance hardware simplicity, speed, and map quality. Most systems excel in one or two of the aforementioned aspects but rarely achieve all. A key issue is the difficulty of initializing 3D Gaussians while concurrently conducting SLAM. To address these challenges, we present Monocular GSO (MGSO), a novel real-time SLAM system that integrates photometric SLAM with 3DGS. Photometric SLAM provides dense structured point clouds for 3DGS initialization, accelerating optimization and producing more efficient maps with fewer Gaussians. As a result, experiments show that our system generates reconstructions with a balance of quality, memory efficiency, and speed that outperforms the state-of-the-art. Furthermore, our system achieves all results using RGB inputs. We evaluate the Replica, TUM-RGBD, and EuRoC datasets against current live dense reconstruction systems. Not only do we surpass contemporary systems, but experiments also show that we maintain our performance on laptop hardware, making it a practical solution for robotics, A/R, and other real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05099v4">DreamMapping: High-Fidelity Text-to-3D Generation via Variational Distribution Mapping</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 15 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Score Distillation Sampling (SDS) has emerged as a prevalent technique for text-to-3D generation, enabling 3D content creation by distilling view-dependent information from text-to-2D guidance. However, they frequently exhibit shortcomings such as over-saturated color and excess smoothness. In this paper, we conduct a thorough analysis of SDS and refine its formulation, finding that the core design is to model the distribution of rendered images. Following this insight, we introduce a novel strategy called Variational Distribution Mapping (VDM), which expedites the distribution modeling process by regarding the rendered images as instances of degradation from diffusion-based generation. This special design enables the efficient training of variational distribution by skipping the calculations of the Jacobians in the diffusion U-Net. We also introduce timestep-dependent Distribution Coefficient Annealing (DCA) to further improve distilling precision. Leveraging VDM and DCA, we use Gaussian Splatting as the 3D representation and build a text-to-3D generation framework. Extensive experiments and evaluations demonstrate the capability of VDM and DCA to generate high-fidelity and realistic assets with optimization efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12899v1">LI-GS: Gaussian Splatting with LiDAR Incorporated for Accurate Large-Scale Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Large-scale 3D reconstruction is critical in the field of robotics, and the potential of 3D Gaussian Splatting (3DGS) for achieving accurate object-level reconstruction has been demonstrated. However, ensuring geometric accuracy in outdoor and unbounded scenes remains a significant challenge. This study introduces LI-GS, a reconstruction system that incorporates LiDAR and Gaussian Splatting to enhance geometric accuracy in large-scale scenes. 2D Gaussain surfels are employed as the map representation to enhance surface alignment. Additionally, a novel modeling method is proposed to convert LiDAR point clouds to plane-constrained multimodal Gaussian Mixture Models (GMMs). The GMMs are utilized during both initialization and optimization stages to ensure sufficient and continuous supervision over the entire scene while mitigating the risk of over-fitting. Furthermore, GMMs are employed in mesh extraction to eliminate artifacts and improve the overall geometric quality. Experiments demonstrate that our method outperforms state-of-the-art methods in large-scale 3D reconstruction, achieving higher accuracy compared to both LiDAR-based methods and Gaussian-based methods with improvements of 52.6% and 68.7%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12892v1">3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 project page: https://lukashoel.github.io/3DGS-LM, video: https://www.youtube.com/watch?v=tDiGuGMssg8, code: https://github.com/lukasHoel/3DGS-LM
    </div>
    <details class="paper-abstract">
      We present 3DGS-LM, a new method that accelerates the reconstruction of 3D Gaussian Splatting (3DGS) by replacing its ADAM optimizer with a tailored Levenberg-Marquardt (LM). Existing methods reduce the optimization time by decreasing the number of Gaussians or by improving the implementation of the differentiable rasterizer. However, they still rely on the ADAM optimizer to fit Gaussian parameters of a scene in thousands of iterations, which can take up to an hour. To this end, we change the optimizer to LM that runs in conjunction with the 3DGS differentiable rasterizer. For efficient GPU parallization, we propose a caching data structure for intermediate gradients that allows us to efficiently calculate Jacobian-vector products in custom CUDA kernels. In every LM iteration, we calculate update directions from multiple image subsets using these kernels and combine them in a weighted mean. Overall, our method is 30% faster than the original 3DGS while obtaining the same reconstruction quality. Our optimization is also agnostic to other methods that acclerate 3DGS, thus enabling even faster speedups compared to vanilla 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12886v1">EdgeGaussians -- 3D Edge Mapping via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      With their meaningful geometry and their omnipresence in the 3D world, edges are extremely useful primitives in computer vision. 3D edges comprise of lines and curves, and methods to reconstruct them use either multi-view images or point clouds as input. State-of-the-art image-based methods first learn a 3D edge point cloud then fit 3D edges to it. The edge point cloud is obtained by learning a 3D neural implicit edge field from which the 3D edge points are sampled on a specific level set (0 or 1). However, such methods present two important drawbacks: i) it is not realistic to sample points on exact level sets due to float imprecision and training inaccuracies. Instead, they are sampled within a range of levels so the points do not lie accurately on the 3D edges and require further processing. ii) Such implicit representations are computationally expensive and require long training times. In this paper, we address these two limitations and propose a 3D edge mapping that is simpler, more efficient, and preserves accuracy. Our method learns explicitly the 3D edge points and their edge direction hence bypassing the need for point sampling. It casts a 3D edge point as the center of a 3D Gaussian and the edge direction as the principal axis of the Gaussian. Such a representation has the advantage of being not only geometrically meaningful but also compatible with the efficient training optimization defined in Gaussian Splatting. Results show that the proposed method produces edges as accurate and complete as the state-of-the-art while being an order of magnitude faster. Code is released at https://github.com/kunalchelani/EdgeGaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12617v1">CrossRT: A cross platform programming technology for hardware-accelerated ray tracing in CG and CV applications</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      We propose a programming technology that bridges cross-platform compatibility and hardware acceleration in ray tracing applications. Our methodology enables developers to define algorithms while our translator manages implementation specifics for different hardware or APIs. Features include: generating hardware-accelerated code from hardware-agnostic, object-oriented C++ algorithm descriptions; enabling users to define software fallbacks for non-hardware-accelerated CPUs and GPUs; producing GPU programming API-based algorithm implementations resembling manually ported C++ versions. The generated code is editable and readable, allowing for additional hardware acceleration. Our translator supports single megakernel and multiple kernel path tracing implementations without altering the programming model or input source code. Wavefront mode is crucial for NeRF and SDF, ensuring efficient evaluation with multiple kernels. Validation on tasks such as BVH tree build/traversal, ray-surface intersection for SDF, ray-volume intersection for 3D Gaussian Splatting, and complex Path Tracing models showed comparable performance levels to expert-written implementations for GPUs. Our technology outperformed existing Path Tracing implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20104v2">Object-centric Reconstruction and Tracking of Dynamic Unknown Objects using 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Accepted at IEEE International Conference on Space Robotics 2024
    </div>
    <details class="paper-abstract">
      Generalizable perception is one of the pillars of high-level autonomy in space robotics. Estimating the structure and motion of unknown objects in dynamic environments is fundamental for such autonomous systems. Traditionally, the solutions have relied on prior knowledge of target objects, multiple disparate representations, or low-fidelity outputs unsuitable for robotic operations. This work proposes a novel approach to incrementally reconstruct and track a dynamic unknown object using a unified representation -- a set of 3D Gaussian blobs that describe its geometry and appearance. The differentiable 3D Gaussian Splatting framework is adapted to a dynamic object-centric setting. The input to the pipeline is a sequential set of RGB-D images. 3D reconstruction and 6-DoF pose tracking tasks are tackled using first-order gradient-based optimization. The formulation is simple, requires no pre-training, assumes no prior knowledge of the object or its motion, and is suitable for online applications. The proposed approach is validated on a dataset of 10 unknown spacecraft of diverse geometry and texture under arbitrary relative motion. The experiments demonstrate successful 3D reconstruction and accurate 6-DoF tracking of the target object in proximity operations over a short to medium duration. The causes of tracking drift are discussed and potential solutions are outlined.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12323v1">Depth Estimation Based on 3D Gaussian Splatting Siamese Defocus</a></div>
    <div class="paper-meta">
      📅 2024-09-18
    </div>
    <details class="paper-abstract">
      Depth estimation is a fundamental task in 3D geometry. While stereo depth estimation can be achieved through triangulation methods, it is not as straightforward for monocular methods, which require the integration of global and local information. The Depth from Defocus (DFD) method utilizes camera lens models and parameters to recover depth information from blurred images and has been proven to perform well. However, these methods rely on All-In-Focus (AIF) images for depth estimation, which is nearly impossible to obtain in real-world applications. To address this issue, we propose a self-supervised framework based on 3D Gaussian splatting and Siamese networks. By learning the blur levels at different focal distances of the same scene in the focal stack, the framework predicts the defocus map and Circle of Confusion (CoC) from a single defocused image, using the defocus map as input to DepthNet for monocular depth estimation. The 3D Gaussian splatting model renders defocused images using the predicted CoC, and the differences between these and the real defocused images provide additional supervision signals for the Siamese Defocus self-supervised network. This framework has been validated on both artificially synthesized and real blurred datasets. Subsequent quantitative and visualization experiments demonstrate that our proposed framework is highly effective as a DFD method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12193v1">Vista3D: Unravel the 3D Darkside of a Single Image</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 ECCV'2024
    </div>
    <details class="paper-abstract">
      We embark on the age-old quest: unveiling the hidden dimensions of objects from mere glimpses of their visible parts. To address this, we present Vista3D, a framework that realizes swift and consistent 3D generation within a mere 5 minutes. At the heart of Vista3D lies a two-phase approach: the coarse phase and the fine phase. In the coarse phase, we rapidly generate initial geometry with Gaussian Splatting from a single image. In the fine phase, we extract a Signed Distance Function (SDF) directly from learned Gaussian Splatting, optimizing it with a differentiable isosurface representation. Furthermore, it elevates the quality of generation by using a disentangled representation with two independent implicit functions to capture both visible and obscured aspects of objects. Additionally, it harmonizes gradients from 2D diffusion prior with 3D-aware diffusion priors by angular diffusion prior composition. Through extensive evaluation, we demonstrate that Vista3D effectively sustains a balance between the consistency and diversity of the generated 3D objects. Demos and code will be available at https://github.com/florinshen/Vista3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03456v2">LM-Gaussian: Boost Sparse-view 3D Gaussian Splatting with Large Model Priors</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Project page: https://hanyangyu1021.github.io/lm-gaussian.github.io/
    </div>
    <details class="paper-abstract">
      We aim to address sparse-view reconstruction of a 3D scene by leveraging priors from large-scale vision models. While recent advancements such as 3D Gaussian Splatting (3DGS) have demonstrated remarkable successes in 3D reconstruction, these methods typically necessitate hundreds of input images that densely capture the underlying scene, making them time-consuming and impractical for real-world applications. However, sparse-view reconstruction is inherently ill-posed and under-constrained, often resulting in inferior and incomplete outcomes. This is due to issues such as failed initialization, overfitting on input images, and a lack of details. To mitigate these challenges, we introduce LM-Gaussian, a method capable of generating high-quality reconstructions from a limited number of images. Specifically, we propose a robust initialization module that leverages stereo priors to aid in the recovery of camera poses and the reliable point clouds. Additionally, a diffusion-based refinement is iteratively applied to incorporate image diffusion priors into the Gaussian optimization process to preserve intricate scene details. Finally, we utilize video diffusion priors to further enhance the rendered images for realistic visual effects. Overall, our approach significantly reduces the data acquisition requirements compared to previous 3DGS methods. We validate the effectiveness of our framework through experiments on various public datasets, demonstrating its potential for high-quality 360-degree scene reconstruction. Visual results are on our website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09748v3">LetsGo: Large-Scale Garage Modeling and Rendering via LiDAR-Assisted Gaussian Primitives</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Project Page: https://zhaofuq.github.io/LetsGo/
    </div>
    <details class="paper-abstract">
      Large garages are ubiquitous yet intricate scenes that present unique challenges due to their monotonous colors, repetitive patterns, reflective surfaces, and transparent vehicle glass. Conventional Structure from Motion (SfM) methods for camera pose estimation and 3D reconstruction often fail in these environments due to poor correspondence construction. To address these challenges, we introduce LetsGo, a LiDAR-assisted Gaussian splatting framework for large-scale garage modeling and rendering. We develop a handheld scanner, Polar, equipped with IMU, LiDAR, and a fisheye camera, to facilitate accurate data acquisition. Using this Polar device, we present the GarageWorld dataset, consisting of eight expansive garage scenes with diverse geometric structures, which will be made publicly available for further research. Our approach demonstrates that LiDAR point clouds collected by the Polar device significantly enhance a suite of 3D Gaussian splatting algorithms for garage scene modeling and rendering. We introduce a novel depth regularizer that effectively eliminates floating artifacts in rendered images. Additionally, we propose a multi-resolution 3D Gaussian representation designed for Level-of-Detail (LOD) rendering. This includes adapted scaling factors for individual levels and a random-resolution-level training scheme to optimize the Gaussians across different resolutions. This representation enables efficient rendering of large-scale garage scenes on lightweight devices via a web-based renderer. Experimental results on our GarageWorld dataset, as well as on ScanNet++ and KITTI-360, demonstrate the superiority of our method in terms of rendering quality and resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11577v2">3DGS-Calib: 3D Gaussian Splatting for Multimodal SpatioTemporal Calibration</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Accepted at IROS 2024 (Oral presentation). Project page: https://qherau.github.io/3DGS-Calib/
    </div>
    <details class="paper-abstract">
      Reliable multimodal sensor fusion algorithms require accurate spatiotemporal calibration. Recently, targetless calibration techniques based on implicit neural representations have proven to provide precise and robust results. Nevertheless, such methods are inherently slow to train given the high computational overhead caused by the large number of sampled points required for volume rendering. With the recent introduction of 3D Gaussian Splatting as a faster alternative to implicit representation methods, we propose to leverage this new rendering approach to achieve faster multi-sensor calibration. We introduce 3DGS-Calib, a new calibration method that relies on the speed and rendering accuracy of 3D Gaussian Splatting to achieve multimodal spatiotemporal calibration that is accurate, robust, and with a substantial speed-up compared to methods relying on implicit neural representations. We demonstrate the superiority of our proposal with experimental results on sequences from KITTI-360, a widely used driving dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11681v1">Gradient-Driven 3D Segmentation and Affordance Transfer in Gaussian Splatting Using 2D Masks</a></div>
    <div class="paper-meta">
      📅 2024-09-18
      | 💬 Preprint, Under review for ICRA 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as a powerful 3D scene representation technique, capturing fine details with high efficiency. In this paper, we introduce a novel voting-based method that extends 2D segmentation models to 3D Gaussian splats. Our approach leverages masked gradients, where gradients are filtered by input 2D masks, and these gradients are used as votes to achieve accurate segmentation. As a byproduct, we discovered that inference-time gradients can also be used to prune Gaussians, resulting in up to 21% compression. Additionally, we explore few-shot affordance transfer, allowing annotations from 2D images to be effectively transferred onto 3D Gaussian splats. The robust yet straightforward mathematical formulation underlying this approach makes it a highly effective tool for numerous downstream applications, such as augmented reality (AR), object editing, and robotics. The project code and additional resources are available at https://jojijoseph.github.io/3dgs-segmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11356v1">RenderWorld: World Model with Self-Supervised 3D Label</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      End-to-end autonomous driving with vision-only is not only more cost-effective compared to LiDAR-vision fusion but also more reliable than traditional methods. To achieve a economical and robust purely visual autonomous driving system, we propose RenderWorld, a vision-only end-to-end autonomous driving framework, which generates 3D occupancy labels using a self-supervised gaussian-based Img2Occ Module, then encodes the labels by AM-VAE, and uses world model for forecasting and planning. RenderWorld employs Gaussian Splatting to represent 3D scenes and render 2D images greatly improves segmentation accuracy and reduces GPU memory consumption compared with NeRF-based methods. By applying AM-VAE to encode air and non-air separately, RenderWorld achieves more fine-grained scene element representation, leading to state-of-the-art performance in both 4D occupancy forecasting and motion planning from autoregressive world model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11307v1">GS-Net: Generalizable Plug-and-Play 3D Gaussian Splatting Module</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) integrates the strengths of primitive-based representations and volumetric rendering techniques, enabling real-time, high-quality rendering. However, 3DGS models typically overfit to single-scene training and are highly sensitive to the initialization of Gaussian ellipsoids, heuristically derived from Structure from Motion (SfM) point clouds, which limits both generalization and practicality. To address these limitations, we propose GS-Net, a generalizable, plug-and-play 3DGS module that densifies Gaussian ellipsoids from sparse SfM point clouds, enhancing geometric structure representation. To the best of our knowledge, GS-Net is the first plug-and-play 3DGS module with cross-scene generalization capabilities. Additionally, we introduce the CARLA-NVS dataset, which incorporates additional camera viewpoints to thoroughly evaluate reconstruction and rendering quality. Extensive experiments demonstrate that applying GS-Net to 3DGS yields a PSNR improvement of 2.08 dB for conventional viewpoints and 1.86 dB for novel viewpoints, confirming the method's effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14873v2">Robo-GS: A Physics Consistent Spatial-Temporal Model for Robotic Arm with Hybrid Representation</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      Real2Sim2Real plays a critical role in robotic arm control and reinforcement learning, yet bridging this gap remains a significant challenge due to the complex physical properties of robots and the objects they manipulate. Existing methods lack a comprehensive solution to accurately reconstruct real-world objects with spatial representations and their associated physics attributes. We propose a Real2Sim pipeline with a hybrid representation model that integrates mesh geometry, 3D Gaussian kernels, and physics attributes to enhance the digital asset representation of robotic arms. This hybrid representation is implemented through a Gaussian-Mesh-Pixel binding technique, which establishes an isomorphic mapping between mesh vertices and Gaussian models. This enables a fully differentiable rendering pipeline that can be optimized through numerical solvers, achieves high-fidelity rendering via Gaussian Splatting, and facilitates physically plausible simulation of the robotic arm's interaction with its environment using mesh-based methods. The code,full presentation and datasets will be made publicly available at our website https://robostudioapp.com
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11211v1">SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 ECCV 2024 paper. The project page and code are available at https://markomih.github.io/SplatFields/
    </div>
    <details class="paper-abstract">
      Digitizing 3D static scenes and 4D dynamic events from multi-view images has long been a challenge in computer vision and graphics. Recently, 3D Gaussian Splatting (3DGS) has emerged as a practical and scalable reconstruction method, gaining popularity due to its impressive reconstruction quality, real-time rendering capabilities, and compatibility with widely used visualization tools. However, the method requires a substantial number of input views to achieve high-quality scene reconstruction, introducing a significant practical bottleneck. This challenge is especially severe in capturing dynamic scenes, where deploying an extensive camera array can be prohibitively costly. In this work, we identify the lack of spatial autocorrelation of splat features as one of the factors contributing to the suboptimal performance of the 3DGS technique in sparse reconstruction settings. To address the issue, we propose an optimization strategy that effectively regularizes splat features by modeling them as the outputs of a corresponding implicit neural field. This results in a consistent enhancement of reconstruction quality across various scenarios. Our approach effectively handles static and dynamic cases, as demonstrated by extensive testing across different setups and scene complexities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08947v2">A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 Project site https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/
    </div>
    <details class="paper-abstract">
      Relighting radiance fields is severely underconstrained for multi-view data, which is most often captured under a single illumination condition; It is especially hard for full scenes containing multiple objects. We introduce a method to create relightable radiance fields using such single-illumination data by exploiting priors extracted from 2D image diffusion models. We first fine-tune a 2D diffusion model on a multi-illumination dataset conditioned by light direction, allowing us to augment a single-illumination capture into a realistic -- but possibly inconsistent -- multi-illumination dataset from directly defined light directions. We use this augmented data to create a relightable radiance field represented by 3D Gaussian splats. To allow direct control of light direction for low-frequency lighting, we represent appearance with a multi-layer perceptron parameterized on light direction. To enforce multi-view consistency and overcome inaccuracies we optimize a per-image auxiliary feature vector. We show results on synthetic and real multi-view data under single illumination, demonstrating that our method successfully exploits 2D diffusion model priors to allow realistic 3D relighting for complete scenes. Project site https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10982v1">GLC-SLAM: Gaussian Splatting SLAM with Efficient Loop Closure</a></div>
    <div class="paper-meta">
      📅 2024-09-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention for its application in dense Simultaneous Localization and Mapping (SLAM), enabling real-time rendering and high-fidelity mapping. However, existing 3DGS-based SLAM methods often suffer from accumulated tracking errors and map drift, particularly in large-scale environments. To address these issues, we introduce GLC-SLAM, a Gaussian Splatting SLAM system that integrates global optimization of camera poses and scene models. Our approach employs frame-to-model tracking and triggers hierarchical loop closure using a global-to-local strategy to minimize drift accumulation. By dividing the scene into 3D Gaussian submaps, we facilitate efficient map updates following loop corrections in large scenes. Additionally, our uncertainty-minimized keyframe selection strategy prioritizes keyframes observing more valuable 3D Gaussians to enhance submap optimization. Experimental results on various datasets demonstrate that GLC-SLAM achieves superior or competitive tracking and mapping performance compared to state-of-the-art dense RGB-D SLAM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.08669v1">AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius</a></div>
    <div class="paper-meta">
      📅 2024-09-17
      | 💬 SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 03-06, 2024, Tokyo, Japan
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a recent explicit 3D representation that has achieved high-quality reconstruction and real-time rendering of complex scenes. However, the rasterization pipeline still suffers from unnecessary overhead resulting from avoidable serial Gaussian culling, and uneven load due to the distinct number of Gaussian to be rendered across pixels, which hinders wider promotion and application of 3DGS. In order to accelerate Gaussian splatting, we propose AdR-Gaussian, which moves part of serial culling in Render stage into the earlier Preprocess stage to enable parallel culling, employing adaptive radius to narrow the rendering pixel range for each Gaussian, and introduces a load balancing method to minimize thread waiting time during the pixel-parallel rendering. Our contributions are threefold, achieving a rendering speed of 310% while maintaining equivalent or even better quality than the state-of-the-art. Firstly, we propose to early cull Gaussian-Tile pairs of low splatting opacity based on an adaptive radius in the Gaussian-parallel Preprocess stage, which reduces the number of affected tile through the Gaussian bounding circle, thus reducing unnecessary overhead and achieving faster rendering speed. Secondly, we further propose early culling based on axis-aligned bounding box for Gaussian splatting, which achieves a more significant reduction in ineffective expenses by accurately calculating the Gaussian size in the 2D directions. Thirdly, we propose a balancing algorithm for pixel thread load, which compresses the information of heavy-load pixels to reduce thread waiting time, and enhance information of light-load pixels to hedge against rendering quality loss. Experiments on three datasets demonstrate that our algorithm can significantly improve the Gaussian Splatting rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10335v1">Phys3DGS: Physically-based 3D Gaussian Splatting for Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2024-09-16
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      We propose two novel ideas (adoption of deferred rendering and mesh-based representation) to improve the quality of 3D Gaussian splatting (3DGS) based inverse rendering. We first report a problem incurred by hidden Gaussians, where Gaussians beneath the surface adversely affect the pixel color in the volume rendering adopted by the existing methods. In order to resolve the problem, we propose applying deferred rendering and report new problems incurred in a naive application of deferred rendering to the existing 3DGS-based inverse rendering. In an effort to improve the quality of 3DGS-based inverse rendering under deferred rendering, we propose a novel two-step training approach which (1) exploits mesh extraction and utilizes a hybrid mesh-3DGS representation and (2) applies novel regularization methods to better exploit the mesh. Our experiments show that, under relighting, the proposed method offers significantly better rendering quality than the existing 3DGS-based inverse rendering methods. Compared with the SOTA voxel grid-based inverse rendering method, it gives better rendering quality while offering real-time rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10216v1">BEINGS: Bayesian Embodied Image-goal Navigation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Image-goal navigation enables a robot to reach the location where a target image was captured, using visual cues for guidance. However, current methods either rely heavily on data and computationally expensive learning-based approaches or lack efficiency in complex environments due to insufficient exploration strategies. To address these limitations, we propose Bayesian Embodied Image-goal Navigation Using Gaussian Splatting, a novel method that formulates ImageNav as an optimal control problem within a model predictive control framework. BEINGS leverages 3D Gaussian Splatting as a scene prior to predict future observations, enabling efficient, real-time navigation decisions grounded in the robot's sensory experiences. By integrating Bayesian updates, our method dynamically refines the robot's strategy without requiring extensive prior experience or data. Our algorithm is validated through extensive simulations and physical experiments, showcasing its potential for embodied robot systems in visually complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10101v1">Adaptive Segmentation-Based Initialization for Steered Mixture of Experts Image Regression</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      Kernel image regression methods have shown to provide excellent efficiency in many image processing task, such as image and light-field compression, Gaussian Splatting, denoising and super-resolution. The estimation of parameters for these methods frequently employ gradient descent iterative optimization, which poses significant computational burden for many applications. In this paper, we introduce a novel adaptive segmentation-based initialization method targeted for optimizing Steered-Mixture-of Experts (SMoE) gating networks and Radial-Basis-Function (RBF) networks with steering kernels. The novel initialization method allocates kernels into pre-calculated image segments. The optimal number of kernels, kernel positions, and steering parameters are derived per segment in an iterative optimization and kernel sparsification procedure. The kernel information from "local" segments is then transferred into a "global" initialization, ready for use in iterative optimization of SMoE, RBF, and related kernel image regression methods. Results show that drastic objective and subjective quality improvements are achievable compared to widely used regular grid initialization, "state-of-the-art" K-Means initialization and previously introduced segmentation-based initialization methods, while also drastically improving the sparsity of the regression models. For same quality, the novel initialization results in models with around 50% reduction of kernels. In addition, a significant reduction of convergence time is achieved, with overall run-time savings of up to 50%. The segmentation-based initialization strategy itself admits heavy parallel computation; in theory, it may be divided into as many tasks as there are segments in the images. By accessing only four parallel GPUs, run-time savings of already 50% for initialization are achievable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10041v1">DENSER: 3D Gaussians Splatting for Scene Reconstruction of Dynamic Urban Environments</a></div>
    <div class="paper-meta">
      📅 2024-09-16
    </div>
    <details class="paper-abstract">
      This paper presents DENSER, an efficient and effective approach leveraging 3D Gaussian splatting (3DGS) for the reconstruction of dynamic urban environments. While several methods for photorealistic scene representations, both implicitly using neural radiance fields (NeRF) and explicitly using 3DGS have shown promising results in scene reconstruction of relatively complex dynamic scenes, modeling the dynamic appearance of foreground objects tend to be challenging, limiting the applicability of these methods to capture subtleties and details of the scenes, especially far dynamic objects. To this end, we propose DENSER, a framework that significantly enhances the representation of dynamic objects and accurately models the appearance of dynamic objects in the driving scene. Instead of directly using Spherical Harmonics (SH) to model the appearance of dynamic objects, we introduce and integrate a new method aiming at dynamically estimating SH bases using wavelets, resulting in better representation of dynamic objects appearance in both space and time. Besides object appearance, DENSER enhances object shape representation through densification of its point cloud across multiple scene frames, resulting in faster convergence of model training. Extensive evaluations on KITTI dataset show that the proposed approach significantly outperforms state-of-the-art methods by a wide margin. Source codes and models will be uploaded to this repository https://github.com/sntubix/denser
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09868v1">SAFER-Splat: A Control Barrier Function for Safe Navigation with Online Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2024-09-15
    </div>
    <details class="paper-abstract">
      SAFER-Splat (Simultaneous Action Filtering and Environment Reconstruction) is a real-time, scalable, and minimally invasive action filter, based on control barrier functions, for safe robotic navigation in a detailed map constructed at runtime using Gaussian Splatting (GSplat). We propose a novel Control Barrier Function (CBF) that not only induces safety with respect to all Gaussian primitives in the scene, but when synthesized into a controller, is capable of processing hundreds of thousands of Gaussians while maintaining a minimal memory footprint and operating at 15 Hz during online Splat training. Of the total compute time, a small fraction of it consumes GPU resources, enabling uninterrupted training. The safety layer is minimally invasive, correcting robot actions only when they are unsafe. To showcase the safety filter, we also introduce SplatBridge, an open-source software package built with ROS for real-time GSplat mapping for robots. We demonstrate the safety and robustness of our pipeline first in simulation, where our method is 20-50x faster, safer, and less conservative than competing methods based on neural radiance fields. Further, we demonstrate simultaneous GSplat mapping and safety filtering on a drone hardware platform using only on-board perception. We verify that under teleoperation a human pilot cannot invoke a collision. Our videos and codebase can be found at https://chengine.github.io/safer-splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09756v1">MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute Transformation</a></div>
    <div class="paper-meta">
      📅 2024-09-15
      | 💬 18 pages, 8 figures, ECCV 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting demonstrates excellent quality and speed in novel view synthesis. Nevertheless, the huge file size of the 3D Gaussians presents challenges for transmission and storage. Current works design compact models to replace the substantial volume and attributes of 3D Gaussians, along with intensive training to distill information. These endeavors demand considerable training time, presenting formidable hurdles for practical deployment. To this end, we propose MesonGS, a codec for post-training compression of 3D Gaussians. Initially, we introduce a measurement criterion that considers both view-dependent and view-independent factors to assess the impact of each Gaussian point on the rendering output, enabling the removal of insignificant points. Subsequently, we decrease the entropy of attributes through two transformations that complement subsequent entropy coding techniques to enhance the file compression rate. More specifically, we first replace rotation quaternions with Euler angles; then, we apply region adaptive hierarchical transform to key attributes to reduce entropy. Lastly, we adopt finer-grained quantization to avoid excessive information loss. Moreover, a well-crafted finetune scheme is devised to restore quality. Extensive experiments demonstrate that MesonGS significantly reduces the size of 3D Gaussians while preserving competitive quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19331v2">NPGA: Neural Parametric Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 Project Page: see https://simongiebenhain.github.io/NPGA/ ; Youtube Video: see https://youtu.be/t0S0OK7WnA4
    </div>
    <details class="paper-abstract">
      The creation of high-fidelity, digital versions of human heads is an important stepping stone in the process of further integrating virtual components into our everyday lives. Constructing such avatars is a challenging research problem, due to a high demand for photo-realism and real-time rendering performance. In this work, we propose Neural Parametric Gaussian Avatars (NPGA), a data-driven approach to create high-fidelity, controllable avatars from multi-view video recordings. We build our method around 3D Gaussian splatting for its highly efficient rendering and to inherit the topological flexibility of point clouds. In contrast to previous work, we condition our avatars' dynamics on the rich expression space of neural parametric head models (NPHM), instead of mesh-based 3DMMs. To this end, we distill the backward deformation field of our underlying NPHM into forward deformations which are compatible with rasterization-based rendering. All remaining fine-scale, expression-dependent details are learned from the multi-view videos. For increased representational capacity of our avatars, we propose per-Gaussian latent features that condition each primitives dynamic behavior. To regularize this increased dynamic expressivity, we propose Laplacian terms on the latent features and predicted dynamics. We evaluate our method on the public NeRSemble dataset, demonstrating that NPGA significantly outperforms the previous state-of-the-art avatars on the self-reenactment task by 2.6 PSNR. Furthermore, we demonstrate accurate animation capabilities from real-world monocular videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08669v1">AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius</a></div>
    <div class="paper-meta">
      📅 2024-09-13
      | 💬 SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 03-06, 2024, Tokyo, Japan
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a recent explicit 3D representation that has achieved high-quality reconstruction and real-time rendering of complex scenes. However, the rasterization pipeline still suffers from unnecessary overhead resulting from avoidable serial Gaussian culling, and uneven load due to the distinct number of Gaussian to be rendered across pixels, which hinders wider promotion and application of 3DGS. In order to accelerate Gaussian splatting, we propose AdR-Gaussian, which moves part of serial culling in Render stage into the earlier Preprocess stage to enable parallel culling, employing adaptive radius to narrow the rendering pixel range for each Gaussian, and introduces a load balancing method to minimize thread waiting time during the pixel-parallel rendering. Our contributions are threefold, achieving a rendering speed of 310% while maintaining equivalent or even better quality than the state-of-the-art. Firstly, we propose to early cull Gaussian-Tile pairs of low splatting opacity based on an adaptive radius in the Gaussian-parallel Preprocess stage, which reduces the number of affected tile through the Gaussian bounding circle, thus reducing unnecessary overhead and achieving faster rendering speed. Secondly, we further propose early culling based on axis-aligned bounding box for Gaussian splatting, which achieves a more significant reduction in ineffective expenses by accurately calculating the Gaussian size in the 2D directions. Thirdly, we propose a balancing algorithm for pixel thread load, which compresses the information of heavy-load pixels to reduce thread waiting time, and enhance information of light-load pixels to hedge against rendering quality loss. Experiments on three datasets demonstrate that our algorithm can significantly improve the Gaussian Splatting rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08613v1">Dense Point Clouds Matter: Dust-GS for Scene Reconstruction from Sparse Viewpoints</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in scene synthesis and novel view synthesis tasks. Typically, the initialization of 3D Gaussian primitives relies on point clouds derived from Structure-from-Motion (SfM) methods. However, in scenarios requiring scene reconstruction from sparse viewpoints, the effectiveness of 3DGS is significantly constrained by the quality of these initial point clouds and the limited number of input images. In this study, we present Dust-GS, a novel framework specifically designed to overcome the limitations of 3DGS in sparse viewpoint conditions. Instead of relying solely on SfM, Dust-GS introduces an innovative point cloud initialization technique that remains effective even with sparse input data. Our approach leverages a hybrid strategy that integrates an adaptive depth-based masking technique, thereby enhancing the accuracy and detail of reconstructed scenes. Extensive experiments conducted on several benchmark datasets demonstrate that Dust-GS surpasses traditional 3DGS methods in scenarios with sparse viewpoints, achieving superior scene reconstruction quality with a reduced number of input images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08562v1">CSS: Overcoming Pose and Scene Challenges in Crowd-Sourced 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-13
    </div>
    <details class="paper-abstract">
      We introduce Crowd-Sourced Splatting (CSS), a novel 3D Gaussian Splatting (3DGS) pipeline designed to overcome the challenges of pose-free scene reconstruction using crowd-sourced imagery. The dream of reconstructing historically significant but inaccessible scenes from collections of photographs has long captivated researchers. However, traditional 3D techniques struggle with missing camera poses, limited viewpoints, and inconsistent lighting. CSS addresses these challenges through robust geometric priors and advanced illumination modeling, enabling high-quality novel view synthesis under complex, real-world conditions. Our method demonstrates clear improvements over existing approaches, paving the way for more accurate and flexible applications in AR, VR, and large-scale 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08353v1">Robust Dual Gaussian Splatting for Immersive Human-centric Volumetric Videos</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 Accepted at SIGGRAPH Asia 2024. Project page: https://nowheretrix.github.io/DualGS/
    </div>
    <details class="paper-abstract">
      Volumetric video represents a transformative advancement in visual media, enabling users to freely navigate immersive virtual experiences and narrowing the gap between digital and real worlds. However, the need for extensive manual intervention to stabilize mesh sequences and the generation of excessively large assets in existing workflows impedes broader adoption. In this paper, we present a novel Gaussian-based approach, dubbed \textit{DualGS}, for real-time and high-fidelity playback of complex human performance with excellent compression ratios. Our key idea in DualGS is to separately represent motion and appearance using the corresponding skin and joint Gaussians. Such an explicit disentanglement can significantly reduce motion redundancy and enhance temporal coherence. We begin by initializing the DualGS and anchoring skin Gaussians to joint Gaussians at the first frame. Subsequently, we employ a coarse-to-fine training strategy for frame-by-frame human performance modeling. It includes a coarse alignment phase for overall motion prediction as well as a fine-grained optimization for robust tracking and high-fidelity rendering. To integrate volumetric video seamlessly into VR environments, we efficiently compress motion using entropy encoding and appearance using codec compression coupled with a persistent codebook. Our approach achieves a compression ratio of up to 120 times, only requiring approximately 350KB of storage per frame. We demonstrate the efficacy of our representation through photo-realistic, free-view experiences on VR headsets, enabling users to immersively watch musicians in performance and feel the rhythm of the notes at the performers' fingertips.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08270v1">FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 ECCV'2024
    </div>
    <details class="paper-abstract">
      This study addresses the challenge of accurately segmenting 3D Gaussian Splatting from 2D masks. Conventional methods often rely on iterative gradient descent to assign each Gaussian a unique label, leading to lengthy optimization and sub-optimal solutions. Instead, we propose a straightforward yet globally optimal solver for 3D-GS segmentation. The core insight of our method is that, with a reconstructed 3D-GS scene, the rendering of the 2D masks is essentially a linear function with respect to the labels of each Gaussian. As such, the optimal label assignment can be solved via linear programming in closed form. This solution capitalizes on the alpha blending characteristic of the splatting process for single step optimization. By incorporating the background bias in our objective function, our method shows superior robustness in 3D segmentation against noises. Remarkably, our optimization completes within 30 seconds, about 50$\times$ faster than the best existing methods. Extensive experiments demonstrate the efficiency and robustness of our method in segmenting various scenes, and its superior performance in downstream tasks such as object removal and inpainting. Demos and code will be available at https://github.com/florinshen/FlashSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08042v1">Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 17 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Novel-view synthesis based on visible light has been extensively studied. In comparison to visible light imaging, thermal infrared imaging offers the advantage of all-weather imaging and strong penetration, providing increased possibilities for reconstruction in nighttime and adverse weather scenarios. However, thermal infrared imaging is influenced by physical characteristics such as atmospheric transmission effects and thermal conduction, hindering the precise reconstruction of intricate details in thermal infrared scenes, manifesting as issues of floaters and indistinct edge features in synthesized images. To address these limitations, this paper introduces a physics-induced 3D Gaussian splatting method named Thermal3D-GS. Thermal3D-GS begins by modeling atmospheric transmission effects and thermal conduction in three-dimensional media using neural networks. Additionally, a temperature consistency constraint is incorporated into the optimization objective to enhance the reconstruction accuracy of thermal infrared images. Furthermore, to validate the effectiveness of our method, the first large-scale benchmark dataset for this field named Thermal Infrared Novel-view Synthesis Dataset (TI-NSD) is created. This dataset comprises 20 authentic thermal infrared video scenes, covering indoor, outdoor, and UAV(Unmanned Aerial Vehicle) scenarios, totaling 6,664 frames of thermal infrared image data. Based on this dataset, this paper experimentally verifies the effectiveness of Thermal3D-GS. The results indicate that our method outperforms the baseline method with a 3.03 dB improvement in PSNR and significantly addresses the issues of floaters and indistinct edge features present in the baseline method. Our dataset and codebase will be released in \href{https://github.com/mzzcdf/Thermal3DGS}{\textcolor{red}{Thermal3DGS}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18544v2">GS-ROR: 3D Gaussian Splatting for Reflective Object Relighting via SDF Priors</a></div>
    <div class="paper-meta">
      📅 2024-09-12
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown a powerful capability for novel view synthesis due to its detailed expressive ability and highly efficient rendering speed. Unfortunately, creating relightable 3D assets with 3DGS is still problematic, particularly for reflective objects, as its discontinuous representation raises difficulties in constraining geometries. Inspired by previous works, the signed distance field (SDF) can serve as an effective way for geometry regularization. However, a direct incorporation between Gaussians and SDF significantly slows training. To this end, we propose GS-ROR for reflective objects relighting with 3DGS aided by SDF priors. At the core of our method is the mutual supervision of the depth and normal between deferred Gaussians and SDF, which avoids the expensive volume rendering of SDF. Thanks to this mutual supervision, the learned deferred Gaussians are well-constrained with a minimal time cost. As the Gaussians are rendered in a deferred shading mode, while the alpha-blended Gaussians are smooth, individual Gaussians may still be outliers, yielding floater artifacts. Therefore, we further introduce an SDF-aware pruning strategy to remove Gaussian outliers, which are located distant from the surface defined by SDF, avoiding the floater issue. Consequently, our method outperforms the existing Gaussian-based inverse rendering methods in terms of relighting quality. Our method also exhibits competitive relighting quality compared to NeRF-based methods with at most 25% of training time and allows rendering at 200+ frames per second on an RTX4090.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.06765v1">gsplat: An Open-Source Library for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-12
      | 💬 17 pages, 2 figures, JMLR MLOSS
    </div>
    <details class="paper-abstract">
      gsplat is an open-source library designed for training and developing Gaussian Splatting methods. It features a front-end with Python bindings compatible with the PyTorch library and a back-end with highly optimized CUDA kernels. gsplat offers numerous features that enhance the optimization of Gaussian Splatting models, which include optimization improvements for speed, memory, and convergence times. Experimental results demonstrate that gsplat achieves up to 10% less training time and 4x less memory than the original implementation. Utilized in several research projects, gsplat is actively maintained on GitHub. Source code is available at https://github.com/nerfstudio-project/gsplat under Apache License 2.0. We welcome contributions from the open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07456v1">Self-Evolving Depth-Supervised 3D Gaussian Splatting from Rendered Stereo Pairs</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 BMVC 2024. Project page: https://kuis-ai.github.io/StereoGS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) significantly struggles to accurately represent the underlying 3D scene geometry, resulting in inaccuracies and floating artifacts when rendering depth maps. In this paper, we address this limitation, undertaking a comprehensive analysis of the integration of depth priors throughout the optimization process of Gaussian primitives, and present a novel strategy for this purpose. This latter dynamically exploits depth cues from a readily available stereo network, processing virtual stereo pairs rendered by the GS model itself during training and achieving consistent self-improvement of the scene representation. Experimental results on three popular datasets, breaking ground as the first to assess depth accuracy for these models, validate our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07452v1">Hi3D: Pursuing High-Resolution Image-to-3D Generation with Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 ACM Multimedia 2024. Source code is available at \url{https://github.com/yanghb22-fdu/Hi3D-Official}
    </div>
    <details class="paper-abstract">
      Despite having tremendous progress in image-to-3D generation, existing methods still struggle to produce multi-view consistent images with high-resolution textures in detail, especially in the paradigm of 2D diffusion that lacks 3D awareness. In this work, we present High-resolution Image-to-3D model (Hi3D), a new video diffusion based paradigm that redefines a single image to multi-view images as 3D-aware sequential image generation (i.e., orbital video generation). This methodology delves into the underlying temporal consistency knowledge in video diffusion model that generalizes well to geometry consistency across multiple views in 3D generation. Technically, Hi3D first empowers the pre-trained video diffusion model with 3D-aware prior (camera pose condition), yielding multi-view images with low-resolution texture details. A 3D-aware video-to-video refiner is learnt to further scale up the multi-view images with high-resolution texture details. Such high-resolution multi-view images are further augmented with novel views through 3D Gaussian Splatting, which are finally leveraged to obtain high-fidelity meshes via 3D reconstruction. Extensive experiments on both novel view synthesis and single view reconstruction demonstrate that our Hi3D manages to produce superior multi-view consistency images with highly-detailed textures. Source code and data are available at \url{https://github.com/yanghb22-fdu/Hi3D-Official}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07245v1">Single-View 3D Reconstruction via SO(2)-Equivariant Gaussian Sculpting Networks</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 Accepted to RSS 2024 Workshop on Geometric and Algebraic Structure in Robot Learning
    </div>
    <details class="paper-abstract">
      This paper introduces SO(2)-Equivariant Gaussian Sculpting Networks (GSNs) as an approach for SO(2)-Equivariant 3D object reconstruction from single-view image observations. GSNs take a single observation as input to generate a Gaussian splat representation describing the observed object's geometry and texture. By using a shared feature extractor before decoding Gaussian colors, covariances, positions, and opacities, GSNs achieve extremely high throughput (>150FPS). Experiments demonstrate that GSNs can be trained efficiently using a multi-view rendering loss and are competitive, in quality, with expensive diffusion-based reconstruction algorithms. The GSN model is validated on multiple benchmark experiments. Moreover, we demonstrate the potential for GSNs to be used within a robotic manipulation pipeline for object-centric grasping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07200v1">ThermalGaussian: Thermal 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Thermography is especially valuable for the military and other users of surveillance cameras. Some recent methods based on Neural Radiance Fields (NeRF) are proposed to reconstruct the thermal scenes in 3D from a set of thermal and RGB images. However, unlike NeRF, 3D Gaussian splatting (3DGS) prevails due to its rapid training and real-time rendering. In this work, we propose ThermalGaussian, the first thermal 3DGS approach capable of rendering high-quality images in RGB and thermal modalities. We first calibrate the RGB camera and the thermal camera to ensure that both modalities are accurately aligned. Subsequently, we use the registered images to learn the multimodal 3D Gaussians. To prevent the overfitting of any single modality, we introduce several multimodal regularization constraints. We also develop smoothing constraints tailored to the physical characteristics of the thermal modality. Besides, we contribute a real-world dataset named RGBT-Scenes, captured by a hand-hold thermal-infrared camera, facilitating future research on thermal scene reconstruction. We conduct comprehensive experiments to show that ThermalGaussian achieves photorealistic rendering of thermal images and improves the rendering quality of RGB images. With the proposed multimodal regularization constraints, we also reduced the model's storage cost by 90\%. The code and dataset will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10772v2">Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes</a></div>
    <div class="paper-meta">
      📅 2024-09-11
      | 💬 Project page: https://niujinshuchong.github.io/gaussian-opacity-fields
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis results, while allowing the rendering of high-resolution images in real-time. However, leveraging 3D Gaussians for surface reconstruction poses significant challenges due to the explicit and disconnected nature of 3D Gaussians. In this work, we present Gaussian Opacity Fields (GOF), a novel approach for efficient, high-quality, and adaptive surface reconstruction in unbounded scenes. Our GOF is derived from ray-tracing-based volume rendering of 3D Gaussians, enabling direct geometry extraction from 3D Gaussians by identifying its levelset, without resorting to Poisson reconstruction or TSDF fusion as in previous work. We approximate the surface normal of Gaussians as the normal of the ray-Gaussian intersection plane, enabling the application of regularization that significantly enhances geometry. Furthermore, we develop an efficient geometry extraction method utilizing Marching Tetrahedra, where the tetrahedral grids are induced from 3D Gaussians and thus adapt to the scene's complexity. Our evaluations reveal that GOF surpasses existing 3DGS-based methods in surface reconstruction and novel view synthesis. Further, it compares favorably to or even outperforms, neural implicit methods in both quality and speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04751v2">Fisheye-GS: Lightweight and Extensible Gaussian Splatting Module for Fisheye Cameras</a></div>
    <div class="paper-meta">
      📅 2024-09-11
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has garnered attention for its high fidelity and real-time rendering. However, adapting 3DGS to different camera models, particularly fisheye lenses, poses challenges due to the unique 3D to 2D projection calculation. Additionally, there are inefficiencies in the tile-based splatting, especially for the extreme curvature and wide field of view of fisheye lenses, which are crucial for its broader real-life applications. To tackle these challenges, we introduce Fisheye-GS.This innovative method recalculates the projection transformation and its gradients for fisheye cameras. Our approach can be seamlessly integrated as a module into other efficient 3D rendering methods, emphasizing its extensibility, lightweight nature, and modular design. Since we only modified the projection component, it can also be easily adapted for use with different camera models. Compared to methods that train after undistortion, our approach demonstrates a clear improvement in visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18717v2">Dynamic Gaussian Marbles for Novel View Synthesis of Casual Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      Gaussian splatting has become a popular representation for novel-view synthesis, exhibiting clear strengths in efficiency, photometric quality, and compositional edibility. Following its success, many works have extended Gaussians to 4D, showing that dynamic Gaussians maintain these benefits while also tracking scene geometry far better than alternative representations. Yet, these methods assume dense multi-view videos as supervision. In this work, we are interested in extending the capability of Gaussian scene representations to casually captured monocular videos. We show that existing 4D Gaussian methods dramatically fail in this setup because the monocular setting is underconstrained. Building off this finding, we propose a method we call Dynamic Gaussian Marbles, which consist of three core modifications that target the difficulties of the monocular setting. First, we use isotropic Gaussian "marbles'', reducing the degrees of freedom of each Gaussian. Second, we employ a hierarchical divide and-conquer learning strategy to efficiently guide the optimization towards solutions with globally coherent motion. Finally, we add image-level and geometry-level priors into the optimization, including a tracking loss that takes advantage of recent progress in point tracking. By constraining the optimization, Dynamic Gaussian Marbles learns Gaussian trajectories that enable novel-view rendering and accurately capture the 3D motion of the scene elements. We evaluate on the Nvidia Dynamic Scenes dataset and the DyCheck iPhone dataset, and show that Gaussian Marbles significantly outperforms other Gaussian baselines in quality, and is on-par with non-Gaussian representations, all while maintaining the efficiency, compositionality, editability, and tracking benefits of Gaussians. Our project page can be found here https://geometry.stanford.edu/projects/dynamic-gaussian-marbles.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06765v1">gsplat: An Open-Source Library for Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 17 pages, 2 figures, JMLR MLOSS
    </div>
    <details class="paper-abstract">
      gsplat is an open-source library designed for training and developing Gaussian Splatting methods. It features a front-end with Python bindings compatible with the PyTorch library and a back-end with highly optimized CUDA kernels. gsplat offers numerous features that enhance the optimization of Gaussian Splatting models, which include optimization improvements for speed, memory, and convergence times. Experimental results demonstrate that gsplat achieves up to 10% less training time and 4x less memory than the original implementation. Utilized in several research projects, gsplat is actively maintained on GitHub. Source code is available at https://github.com/nerfstudio-project/gsplat under Apache License 2.0. We welcome contributions from the open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06685v1">GigaGS: Scaling up Planar-Based 3D Gaussians for Large Scene Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-10
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown promising performance in novel view synthesis. Previous methods adapt it to obtaining surfaces of either individual 3D objects or within limited scenes. In this paper, we make the first attempt to tackle the challenging task of large-scale scene surface reconstruction. This task is particularly difficult due to the high GPU memory consumption, different levels of details for geometric representation, and noticeable inconsistencies in appearance. To this end, we propose GigaGS, the first work for high-quality surface reconstruction for large-scale scenes using 3DGS. GigaGS first applies a partitioning strategy based on the mutual visibility of spatial regions, which effectively grouping cameras for parallel processing. To enhance the quality of the surface, we also propose novel multi-view photometric and geometric consistency constraints based on Level-of-Detail representation. In doing so, our method can reconstruct detailed surface structures. Comprehensive experiments are conducted on various datasets. The consistent improvement demonstrates the superiority of GigaGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06407v1">Sources of Uncertainty in 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 To appear in ECCV 2024 Workshop Proceedings. Project page at https://aaltoml.github.io/uncertainty-nerf-gs/
    </div>
    <details class="paper-abstract">
      The process of 3D scene reconstruction can be affected by numerous uncertainty sources in real-world scenes. While Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (GS) achieve high-fidelity rendering, they lack built-in mechanisms to directly address or quantify uncertainties arising from the presence of noise, occlusions, confounding outliers, and imprecise camera pose inputs. In this paper, we introduce a taxonomy that categorizes different sources of uncertainty inherent in these methods. Moreover, we extend NeRF- and GS-based methods with uncertainty estimation techniques, including learning uncertainty outputs and ensembles, and perform an empirical study to assess their ability to capture the sensitivity of the reconstruction. Our study highlights the need for addressing various uncertainty aspects when designing NeRF/GS-based methods for uncertainty-aware 3D reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06241v2">MGS-SLAM: Monocular Sparse Tracking and Gaussian Mapping with Depth Smooth Regularization</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 Accepted by IEEE Robotics and Automation Letters
    </div>
    <details class="paper-abstract">
      This letter introduces a novel framework for dense Visual Simultaneous Localization and Mapping (VSLAM) based on Gaussian Splatting. Recently, SLAM based on Gaussian Splatting has shown promising results. However, in monocular scenarios, the Gaussian maps reconstructed lack geometric accuracy and exhibit weaker tracking capability. To address these limitations, we jointly optimize sparse visual odometry tracking and 3D Gaussian Splatting scene representation for the first time. We obtain depth maps on visual odometry keyframe windows using a fast Multi-View Stereo (MVS) network for the geometric supervision of Gaussian maps. Furthermore, we propose a depth smooth loss and Sparse-Dense Adjustment Ring (SDAR) to reduce the negative effect of estimated depth maps and preserve the consistency in scale between the visual odometry and Gaussian maps. We have evaluated our system across various synthetic and real-world datasets. The accuracy of our pose estimation surpasses existing methods and achieves state-of-the-art. Additionally, it outperforms previous monocular methods in terms of novel view synthesis and geometric reconstruction fidelities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17872v3">HFGS: 4D Gaussian Splatting with Emphasis on Spatial and Temporal High-Frequency Components for Endoscopic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-09-10
      | 💬 BMVC2024
    </div>
    <details class="paper-abstract">
      Robot-assisted minimally invasive surgery benefits from enhancing dynamic scene reconstruction, as it improves surgical outcomes. While Neural Radiance Fields (NeRF) have been effective in scene reconstruction, their slow inference speeds and lengthy training durations limit their applicability. To overcome these limitations, 3D Gaussian Splatting (3D-GS) based methods have emerged as a recent trend, offering rapid inference capabilities and superior 3D quality. However, these methods still struggle with under-reconstruction in both static and dynamic scenes. In this paper, we propose HFGS, a novel approach for deformable endoscopic reconstruction that addresses these challenges from spatial and temporal frequency perspectives. Our approach incorporates deformation fields to better handle dynamic scenes and introduces Spatial High-Frequency Emphasis Reconstruction (SHF) to minimize discrepancies in spatial frequency spectra between the rendered image and its ground truth. Additionally, we introduce Temporal High-Frequency Emphasis Reconstruction (THF) to enhance dynamic awareness in neural rendering by leveraging flow priors, focusing optimization on motion-intensive parts. Extensive experiments on two widely used benchmarks demonstrate that HFGS achieves superior rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06037v1">Online 3D reconstruction and dense tracking in endoscopic videos</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      3D scene reconstruction from stereo endoscopic video data is crucial for advancing surgical interventions. In this work, we present an online framework for online, dense 3D scene reconstruction and tracking, aimed at enhancing surgical scene understanding and assisting interventions. Our method dynamically extends a canonical scene representation using Gaussian splatting, while modeling tissue deformations through a sparse set of control points. We introduce an efficient online fitting algorithm that optimizes the scene parameters, enabling consistent tracking and accurate reconstruction. Through experiments on the StereoMIS dataset, we demonstrate the effectiveness of our approach, outperforming state-of-the-art tracking methods and achieving comparable performance to offline reconstruction techniques. Our work enables various downstream applications thus contributing to advancing the capabilities of surgical assistance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05819v1">GASP: Gaussian Splatting for Physic-Based Simulations</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      Physics simulation is paramount for modeling and utilization of 3D scenes in various real-world applications. However, its integration with state-of-the-art 3D scene rendering techniques such as Gaussian Splatting (GS) remains challenging. Existing models use additional meshing mechanisms, including triangle or tetrahedron meshing, marching cubes, or cage meshes. As an alternative, we can modify the physics grounded Newtonian dynamics to align with 3D Gaussian components. Current models take the first-order approximation of a deformation map, which locally approximates the dynamics by linear transformations. In contrast, our Gaussian Splatting for Physics-Based Simulations (GASP) model uses such a map (without any modifications) and flat Gaussian distributions, which are parameterized by three points (mesh faces). Subsequently, each 3D point (mesh face node) is treated as a discrete entity within a 3D space. Consequently, the problem of modeling Gaussian components is reduced to working with 3D points. Additionally, the information on mesh faces can be used to incorporate further properties into the physics model, facilitating the use of triangles. Resulting solution can be integrated into any physics engine that can be treated as a black box. As demonstrated in our studies, the proposed model exhibits superior performance on a diverse range of benchmark datasets designed for 3D object rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16296v1">LiDAR-3DGS: LiDAR Reinforced 3D Gaussian Splatting for Multimodal Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2024-09-09
    </div>
    <details class="paper-abstract">
      In this paper, we explore the capabilities of multimodal inputs to 3D Gaussian Splatting (3DGS) based Radiance Field Rendering. We present LiDAR-3DGS, a novel method of reinforcing 3DGS inputs with LiDAR generated point clouds to significantly improve the accuracy and detail of 3D models. We demonstrate a systematic approach of LiDAR reinforcement to 3DGS to enable capturing of important features such as bolts, apertures, and other details that are often missed by image-based features alone. These details are crucial for engineering applications such as remote monitoring and maintenance. Without modifying the underlying 3DGS algorithm, we demonstrate that even a modest addition of LiDAR generated point cloud significantly enhances the perceptual quality of the models. At 30k iterations, the model generated by our method resulted in an increase of 7.064% in PSNR and 0.565% in SSIM, respectively. Since the LiDAR used in this research was a commonly used commercial-grade device, the improvements observed were modest and can be further enhanced with higher-grade LiDAR systems. Additionally, these improvements can be supplementary to other derivative works of Radiance Field Rendering and also provide a new insight for future LiDAR and computer vision integrated modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05334v1">Lagrangian Hashing for Compressed Neural Field Representations</a></div>
    <div class="paper-meta">
      📅 2024-09-09
      | 💬 Project page: https://theialab.github.io/laghashes/
    </div>
    <details class="paper-abstract">
      We present Lagrangian Hashing, a representation for neural fields combining the characteristics of fast training NeRF methods that rely on Eulerian grids (i.e.~InstantNGP), with those that employ points equipped with features as a way to represent information (e.g. 3D Gaussian Splatting or PointNeRF). We achieve this by incorporating a point-based representation into the high-resolution layers of the hierarchical hash tables of an InstantNGP representation. As our points are equipped with a field of influence, our representation can be interpreted as a mixture of Gaussians stored within the hash table. We propose a loss that encourages the movement of our Gaussians towards regions that require more representation budget to be sufficiently well represented. Our main finding is that our representation allows the reconstruction of signals using a more compact representation without compromising quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.08498v2">Gaussian Splatting in Style</a></div>
    <div class="paper-meta">
      📅 2024-09-09
      | 💬 GCPR 2024
    </div>
    <details class="paper-abstract">
      3D scene stylization extends the work of neural style transfer to 3D. A vital challenge in this problem is to maintain the uniformity of the stylized appearance across multiple views. A vast majority of the previous works achieve this by training a 3D model for every stylized image and a set of multi-view images. In contrast, we propose a novel architecture trained on a collection of style images that, at test time, produces real time high-quality stylized novel views. We choose the underlying 3D scene representation for our model as 3D Gaussian splatting. We take the 3D Gaussians and process them using a multi-resolution hash grid and a tiny MLP to obtain stylized views. The MLP is conditioned on different style codes for generalization to different styles during test time. The explicit nature of 3D Gaussians gives us inherent advantages over NeRF-based methods, including geometric consistency and a fast training and rendering regime. This enables our method to be useful for various practical use cases, such as augmented or virtual reality. We demonstrate that our method achieves state-of-the-art performance with superior visual quality on various indoor and outdoor real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19586v2">TOGS: Gaussian Splatting with Temporal Opacity Offset for Real-Time 4D DSA Rendering</a></div>
    <div class="paper-meta">
      📅 2024-09-08
    </div>
    <details class="paper-abstract">
      Four-dimensional Digital Subtraction Angiography (4D DSA) is a medical imaging technique that provides a series of 2D images captured at different stages and angles during the process of contrast agent filling blood vessels. It plays a significant role in the diagnosis of cerebrovascular diseases. Improving the rendering quality and speed under sparse sampling is important for observing the status and location of lesions. The current methods exhibit inadequate rendering quality in sparse views and suffer from slow rendering speed. To overcome these limitations, we propose TOGS, a Gaussian splatting method with opacity offset over time, which can effectively improve the rendering quality and speed of 4D DSA. We introduce an opacity offset table for each Gaussian to model the opacity offsets of the Gaussian, using these opacity-varying Gaussians to model the temporal variations in the radiance of the contrast agent. By interpolating the opacity offset table, the opacity variation of the Gaussian at different time points can be determined. This enables us to render the 2D DSA image at that specific moment. Additionally, we introduced a Smooth loss term in the loss function to mitigate overfitting issues that may arise in the model when dealing with sparse view scenarios. During the training phase, we randomly prune Gaussians, thereby reducing the storage overhead of the model. The experimental results demonstrate that compared to previous methods, this model achieves state-of-the-art render quality under the same number of training views. Additionally, it enables real-time rendering while maintaining low storage overhead. The code is available at https://github.com/hustvl/TOGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04963v1">GS-PT: Exploiting 3D Gaussian Splatting for Comprehensive Point Cloud Understanding via Self-supervised Learning</a></div>
    <div class="paper-meta">
      📅 2024-09-08
    </div>
    <details class="paper-abstract">
      Self-supervised learning of point cloud aims to leverage unlabeled 3D data to learn meaningful representations without reliance on manual annotations. However, current approaches face challenges such as limited data diversity and inadequate augmentation for effective feature learning. To address these challenges, we propose GS-PT, which integrates 3D Gaussian Splatting (3DGS) into point cloud self-supervised learning for the first time. Our pipeline utilizes transformers as the backbone for self-supervised pre-training and introduces novel contrastive learning tasks through 3DGS. Specifically, the transformers aim to reconstruct the masked point cloud. 3DGS utilizes multi-view rendered images as input to generate enhanced point cloud distributions and novel view images, facilitating data augmentation and cross-modal contrastive learning. Additionally, we incorporate features from depth maps. By optimizing these tasks collectively, our method enriches the tri-modal self-supervised learning process, enabling the model to leverage the correlation across 3D point clouds and 2D images from various modalities. We freeze the encoder after pre-training and test the model's performance on multiple downstream tasks. Experimental results indicate that GS-PT outperforms the off-the-shelf self-supervised learning methods on various downstream tasks including 3D object classification, real-world classifications, and few-shot learning and segmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04196v1">GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Reconstructing realistic 3D human models from monocular images has significant applications in creative industries, human-computer interfaces, and healthcare. We base our work on 3D Gaussian Splatting (3DGS), a scene representation composed of a mixture of Gaussians. Predicting such mixtures for a human from a single input image is challenging, as it is a non-uniform density (with a many-to-one relationship with input pixels) with strict physical constraints. At the same time, it needs to be flexible to accommodate a variety of clothes and poses. Our key observation is that the vertices of standardized human meshes (such as SMPL) can provide an adequate density and approximate initial position for Gaussians. We can then train a transformer model to jointly predict comparatively small adjustments to these positions, as well as the other Gaussians' attributes and the SMPL parameters. We show empirically that this combination (using only multi-view supervision) can achieve fast inference of 3D human models from a single image without test-time optimization, expensive diffusion models, or 3D points supervision. We also show that it can improve 3D pose estimation by better fitting human models that account for clothes and other variations. The code is available on the project website https://abdullahamdi.com/gst/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08498v2">Gaussian Splatting in Style</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 GCPR 2024
    </div>
    <details class="paper-abstract">
      3D scene stylization extends the work of neural style transfer to 3D. A vital challenge in this problem is to maintain the uniformity of the stylized appearance across multiple views. A vast majority of the previous works achieve this by training a 3D model for every stylized image and a set of multi-view images. In contrast, we propose a novel architecture trained on a collection of style images that, at test time, produces real time high-quality stylized novel views. We choose the underlying 3D scene representation for our model as 3D Gaussian splatting. We take the 3D Gaussians and process them using a multi-resolution hash grid and a tiny MLP to obtain stylized views. The MLP is conditioned on different style codes for generalization to different styles during test time. The explicit nature of 3D Gaussians gives us inherent advantages over NeRF-based methods, including geometric consistency and a fast training and rendering regime. This enables our method to be useful for various practical use cases, such as augmented or virtual reality. We demonstrate that our method achieves state-of-the-art performance with superior visual quality on various indoor and outdoor real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04013v1">3D-GP-LMVIC: Learning-based Multi-View Image Coding with 3D Gaussian Geometric Priors</a></div>
    <div class="paper-meta">
      📅 2024-09-06
      | 💬 19pages, 8 figures, conference
    </div>
    <details class="paper-abstract">
      Multi-view image compression is vital for 3D-related applications. To effectively model correlations between views, existing methods typically predict disparity between two views on a 2D plane, which works well for small disparities, such as in stereo images, but struggles with larger disparities caused by significant view changes. To address this, we propose a novel approach: learning-based multi-view image coding with 3D Gaussian geometric priors (3D-GP-LMVIC). Our method leverages 3D Gaussian Splatting to derive geometric priors of the 3D scene, enabling more accurate disparity estimation across views within the compression model. Additionally, we introduce a depth map compression model to reduce redundancy in geometric information between views. A multi-view sequence ordering method is also proposed to enhance correlations between adjacent views. Experimental results demonstrate that 3D-GP-LMVIC surpasses both traditional and learning-based methods in performance, while maintaining fast encoding and decoding speed.
    </details>
</div>
