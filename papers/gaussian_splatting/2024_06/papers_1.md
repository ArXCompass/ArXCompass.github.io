# gaussian splatting - 2024_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00316v1">OccFusion: Rendering Occluded Humans with Generative Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2024-06-29
    </div>
    <details class="paper-abstract">
      Most existing human rendering methods require every part of the human to be fully visible throughout the input video. However, this assumption does not hold in real-life settings where obstructions are common, resulting in only partial visibility of the human. Considering this, we present OccFusion, an approach that utilizes efficient 3D Gaussian splatting supervised by pretrained 2D diffusion models for efficient and high-fidelity human rendering. We propose a pipeline consisting of three stages. In the Initialization stage, complete human masks are generated from partial visibility masks. In the Optimization stage, 3D human Gaussians are optimized with additional supervision by Score-Distillation Sampling (SDS) to create a complete geometry of the human. Finally, in the Refinement stage, in-context inpainting is designed to further improve rendering quality on the less observed human body parts. We evaluate OccFusion on ZJU-MoCap and challenging OcMotion sequences and find that it achieves state-of-the-art performance in the rendering of occluded humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05702v6">NGM-SLAM: Gaussian Splatting SLAM with Radiance Field Submap</a></div>
    <div class="paper-meta">
      📅 2024-06-28
      | 💬 9pages, 4 figures
    </div>
    <details class="paper-abstract">
      SLAM systems based on Gaussian Splatting have garnered attention due to their capabilities for rapid real-time rendering and high-fidelity mapping. However, current Gaussian Splatting SLAM systems usually struggle with large scene representation and lack effective loop closure detection. To address these issues, we introduce NGM-SLAM, the first 3DGS based SLAM system that utilizes neural radiance field submaps for progressive scene expression, effectively integrating the strengths of neural radiance fields and 3D Gaussian Splatting. We utilize neural radiance field submaps as supervision and achieve high-quality scene expression and online loop closure adjustments through Gaussian rendering of fused submaps. Our results on multiple real-world scenes and large-scale scene datasets demonstrate that our method can achieve accurate hole filling and high-quality scene expression, supporting monocular, stereo, and RGB-D inputs, and achieving state-of-the-art scene reconstruction and tracking performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19434v1">Lightweight Predictive 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-06-27
      | 💬 Project Page: https://plumpuddings.github.io/LPGS//
    </div>
    <details class="paper-abstract">
      Recent approaches representing 3D objects and scenes using Gaussian splats show increased rendering speed across a variety of platforms and devices. While rendering such representations is indeed extremely efficient, storing and transmitting them is often prohibitively expensive. To represent large-scale scenes, one often needs to store millions of 3D Gaussians, occupying gigabytes of disk space. This poses a very practical limitation, prohibiting widespread adoption.Several solutions have been proposed to strike a balance between disk size and rendering quality, noticeably reducing the visual quality. In this work, we propose a new representation that dramatically reduces the hard drive footprint while featuring similar or improved quality when compared to the standard 3D Gaussian splats. When compared to other compact solutions, ours offers higher quality renderings with significantly reduced storage, being able to efficiently run on a mobile device in real-time. Our key observation is that nearby points in the scene can share similar representations. Hence, only a small ratio of 3D points needs to be stored. We introduce an approach to identify such points which are called parent points. The discarded points called children points along with attributes can be efficiently predicted by tiny MLPs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.19434v1">Lightweight Predictive 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-06-27
      | 💬 Project Page: https://plumpuddings.github.io/LPGS//
    </div>
    <details class="paper-abstract">
      Recent approaches representing 3D objects and scenes using Gaussian splats show increased rendering speed across a variety of platforms and devices. While rendering such representations is indeed extremely efficient, storing and transmitting them is often prohibitively expensive. To represent large-scale scenes, one often needs to store millions of 3D Gaussians, occupying gigabytes of disk space. This poses a very practical limitation, prohibiting widespread adoption.Several solutions have been proposed to strike a balance between disk size and rendering quality, noticeably reducing the visual quality. In this work, we propose a new representation that dramatically reduces the hard drive footprint while featuring similar or improved quality when compared to the standard 3D Gaussian splats. When compared to other compact solutions, ours offers higher quality renderings with significantly reduced storage, being able to efficiently run on a mobile device in real-time. Our key observation is that nearby points in the scene can share similar representations. Hence, only a small ratio of 3D points needs to be stored. We introduce an approach to identify such points which are called parent points. The discarded points called children points along with attributes can be efficiently predicted by tiny MLPs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18533v1">On Scaling Up 3D Gaussian Splatting Training</a></div>
    <div class="paper-meta">
      📅 2024-06-26
      | 💬 Code: https://github.com/nyu-systems/Grendel-GS ; Project page: https://daohanlu.github.io/scaling-up-3dgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly popular for 3D reconstruction due to its superior visual quality and rendering speed. However, 3DGS training currently occurs on a single GPU, limiting its ability to handle high-resolution and large-scale 3D reconstruction tasks due to memory constraints. We introduce Grendel, a distributed system designed to partition 3DGS parameters and parallelize computation across multiple GPUs. As each Gaussian affects a small, dynamic subset of rendered pixels, Grendel employs sparse all-to-all communication to transfer the necessary Gaussians to pixel partitions and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training with multiple views. We explore various optimization hyperparameter scaling strategies and find that a simple sqrt(batch size) scaling rule is highly effective. Evaluations using large-scale, high-resolution scenes show that Grendel enhances rendering quality by scaling up 3DGS parameters across multiple GPUs. On the Rubble dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPUs, compared to a PSNR of 26.28 using 11.2 million Gaussians on a single GPU. Grendel is an open-source project available at: https://github.com/nyu-systems/Grendel-GS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18462v1">GaussianDreamerPro: Text to Manipulable 3D Gaussians with Highly Enhanced Quality</a></div>
    <div class="paper-meta">
      📅 2024-06-26
      | 💬 Project page: https://taoranyi.com/gaussiandreamerpro/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian splatting (3D-GS) has achieved great success in reconstructing and rendering real-world scenes. To transfer the high rendering quality to generation tasks, a series of research works attempt to generate 3D-Gaussian assets from text. However, the generated assets have not achieved the same quality as those in reconstruction tasks. We observe that Gaussians tend to grow without control as the generation process may cause indeterminacy. Aiming at highly enhancing the generation quality, we propose a novel framework named GaussianDreamerPro. The main idea is to bind Gaussians to reasonable geometry, which evolves over the whole generation process. Along different stages of our framework, both the geometry and appearance can be enriched progressively. The final output asset is constructed with 3D Gaussians bound to mesh, which shows significantly enhanced details and quality compared with previous methods. Notably, the generated asset can also be seamlessly integrated into downstream manipulation pipelines, e.g. animation, composition, and simulation etc., greatly promoting its potential in wide applications. Demos are available at https://taoranyi.com/gaussiandreamerpro/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18199v1">GS-Octree: Octree-based 3D Gaussian Splatting for Robust Object-level 3D Reconstruction Under Strong Lighting</a></div>
    <div class="paper-meta">
      📅 2024-06-26
    </div>
    <details class="paper-abstract">
      The 3D Gaussian Splatting technique has significantly advanced the construction of radiance fields from multi-view images, enabling real-time rendering. While point-based rasterization effectively reduces computational demands for rendering, it often struggles to accurately reconstruct the geometry of the target object, especially under strong lighting. To address this challenge, we introduce a novel approach that combines octree-based implicit surface representations with Gaussian splatting. Our method consists of four stages. Initially, it reconstructs a signed distance field (SDF) and a radiance field through volume rendering, encoding them in a low-resolution octree. The initial SDF represents the coarse geometry of the target object. Subsequently, it introduces 3D Gaussians as additional degrees of freedom, which are guided by the SDF. In the third stage, the optimized Gaussians further improve the accuracy of the SDF, allowing it to recover finer geometric details compared to the initial SDF obtained in the first stage. Finally, it adopts the refined SDF to further optimize the 3D Gaussians via splatting, eliminating those that contribute little to visual appearance. Experimental results show that our method, which leverages the distribution of 3D Gaussians with SDFs, reconstructs more accurate geometry, particularly in images with specular highlights caused by strong lighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18198v1">VDG: Vision-Only Dynamic Gaussian for Driving Simulation</a></div>
    <div class="paper-meta">
      📅 2024-06-26
    </div>
    <details class="paper-abstract">
      Dynamic Gaussian splatting has led to impressive scene reconstruction and image synthesis advances in novel views. Existing methods, however, heavily rely on pre-computed poses and Gaussian initialization by Structure from Motion (SfM) algorithms or expensive sensors. For the first time, this paper addresses this issue by integrating self-supervised VO into our pose-free dynamic Gaussian method (VDG) to boost pose and depth initialization and static-dynamic decomposition. Moreover, VDG can work with only RGB image input and construct dynamic scenes at a faster speed and larger scenes compared with the pose-free dynamic view-synthesis method. We demonstrate the robustness of our approach via extensive quantitative and qualitative experiments. Our results show favorable performance over the state-of-the-art dynamic view synthesis methods. Additional video and source code will be posted on our project page at https://3d-aigc.github.io/VDG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11285v2">Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices</a></div>
    <div class="paper-meta">
      📅 2024-06-25
    </div>
    <details class="paper-abstract">
      Interactive photorealistic rendering of 3D anatomy is used in medical education to explain the structure of the human body. It is currently restricted to frontal teaching scenarios, where even with a powerful GPU and high-speed access to a large storage device where the data set is hosted, interactive demonstrations can hardly be achieved. We present the use of novel view synthesis via compressed 3D Gaussian Splatting (3DGS) to overcome this restriction, and to even enable students to perform cinematic anatomy on lightweight and mobile devices. Our proposed pipeline first finds a set of camera poses that captures all potentially seen structures in the data. High-quality images are then generated with path tracing and converted into a compact 3DGS representation, consuming < 70 MB even for data sets of multiple GBs. This allows for real-time photorealistic novel view synthesis that recovers structures up to the voxel resolution and is almost indistinguishable from the path-traced images
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17345v1">NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods</a></div>
    <div class="paper-meta">
      📅 2024-06-25
      | 💬 Web: https://jkulhanek.com/nerfbaselines
    </div>
    <details class="paper-abstract">
      Novel view synthesis is an important problem with many applications, including AR/VR, gaming, and simulations for robotics. With the recent rapid development of Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) methods, it is becoming difficult to keep track of the current state of the art (SoTA) due to methods using different evaluation protocols, codebases being difficult to install and use, and methods not generalizing well to novel 3D scenes. Our experiments support this claim by showing that tiny differences in evaluation protocols of various methods can lead to inconsistent reported metrics. To address these issues, we propose a framework called NerfBaselines, which simplifies the installation of various methods, provides consistent benchmarking tools, and ensures reproducibility. We validate our implementation experimentally by reproducing numbers reported in the original papers. To further improve the accessibility, we release a web platform where commonly used methods are compared on standard benchmarks. Web: https://jkulhanek.com/nerfbaselines
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.01467v2">RaDe-GS: Rasterizing Depth in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-25
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has proven to be highly effective in novel view synthesis, achieving high-quality and real-time rendering. However, its potential for reconstructing detailed 3D shapes has not been fully explored. Existing methods often suffer from limited shape accuracy due to the discrete and unstructured nature of Gaussian splats, which complicates the shape extraction. While recent techniques like 2D GS have attempted to improve shape reconstruction, they often reformulate the Gaussian primitives in ways that reduce both rendering quality and computational efficiency. To address these problems, our work introduces a rasterized approach to render the depth maps and surface normal maps of general 3D Gaussian splats. Our method not only significantly enhances shape reconstruction accuracy but also maintains the computational efficiency intrinsic to Gaussian Splatting. It achieves a Chamfer distance error comparable to NeuraLangelo on the DTU dataset and maintains similar computational efficiency as the original 3D GS methods. Our method is a significant advancement in Gaussian Splatting and can be directly integrated into existing Gaussian Splatting-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17074v1">Reducing the Memory Footprint of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-24
      | 💬 Project website: https://repo-sam.inria.fr/fungraph/reduced_3dgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting provides excellent visual quality for novel view synthesis, with fast training and real-time rendering; unfortunately, the memory requirements of this method for storing and transmission are unreasonably high. We first analyze the reasons for this, identifying three main areas where storage can be reduced: the number of 3D Gaussian primitives used to represent a scene, the number of coefficients for the spherical harmonics used to represent directional radiance, and the precision required to store Gaussian primitive attributes. We present a solution to each of these issues. First, we propose an efficient, resolution-aware primitive pruning approach, reducing the primitive count by half. Second, we introduce an adaptive adjustment method to choose the number of coefficients used to represent directional radiance for each Gaussian primitive, and finally a codebook-based quantization method, together with a half-float representation for further memory reduction. Taken together, these three components result in a 27 reduction in overall size on disk on the standard datasets we tested, along with a 1.7 speedup in rendering speed. We demonstrate our method on standard datasets and show how our solution results in significantly reduced download times when using the method on a mobile device.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16850v1">From Perfect to Noisy World Simulation: Customizable Embodied Multi-modal Perturbations for SLAM Robustness Benchmarking</a></div>
    <div class="paper-meta">
      📅 2024-06-24
      | 💬 50 pages. arXiv admin note: substantial text overlap with arXiv:2402.08125
    </div>
    <details class="paper-abstract">
      Embodied agents require robust navigation systems to operate in unstructured environments, making the robustness of Simultaneous Localization and Mapping (SLAM) models critical to embodied agent autonomy. While real-world datasets are invaluable, simulation-based benchmarks offer a scalable approach for robustness evaluations. However, the creation of a challenging and controllable noisy world with diverse perturbations remains under-explored. To this end, we propose a novel, customizable pipeline for noisy data synthesis, aimed at assessing the resilience of multi-modal SLAM models against various perturbations. The pipeline comprises a comprehensive taxonomy of sensor and motion perturbations for embodied multi-modal (specifically RGB-D) sensing, categorized by their sources and propagation order, allowing for procedural composition. We also provide a toolbox for synthesizing these perturbations, enabling the transformation of clean environments into challenging noisy simulations. Utilizing the pipeline, we instantiate the large-scale Noisy-Replica benchmark, which includes diverse perturbation types, to evaluate the risk tolerance of existing advanced RGB-D SLAM models. Our extensive analysis uncovers the susceptibilities of both neural (NeRF and Gaussian Splatting -based) and non-neural SLAM models to disturbances, despite their demonstrated accuracy in standard benchmarks. Our code is publicly available at https://github.com/Xiaohao-Xu/SLAM-under-Perturbation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16815v1">ClotheDreamer: Text-Guided Garment Generation with 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-06-24
      | 💬 Project Page: https://ggxxii.github.io/clothedreamer
    </div>
    <details class="paper-abstract">
      High-fidelity 3D garment synthesis from text is desirable yet challenging for digital avatar creation. Recent diffusion-based approaches via Score Distillation Sampling (SDS) have enabled new possibilities but either intricately couple with human body or struggle to reuse. We introduce ClotheDreamer, a 3D Gaussian-based method for generating wearable, production-ready 3D garment assets from text prompts. We propose a novel representation Disentangled Clothe Gaussian Splatting (DCGS) to enable separate optimization. DCGS represents clothed avatar as one Gaussian model but freezes body Gaussian splats. To enhance quality and completeness, we incorporate bidirectional SDS to supervise clothed avatar and garment RGBD renderings respectively with pose conditions and propose a new pruning strategy for loose clothing. Our approach can also support custom clothing templates as input. Benefiting from our design, the synthetic 3D garment can be easily applied to virtual try-on and support physically accurate animation. Extensive experiments showcase our method's superior and competitive performance. Our project page is at https://ggxxii.github.io/clothedreamer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01467v2">RaDe-GS: Rasterizing Depth in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-24
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has proven to be highly effective in novel view synthesis, achieving high-quality and real-time rendering. However, its potential for reconstructing detailed 3D shapes has not been fully explored. Existing methods often suffer from limited shape accuracy due to the discrete and unstructured nature of Gaussian splats, which complicates the shape extraction. While recent techniques like 2D GS have attempted to improve shape reconstruction, they often reformulate the Gaussian primitives in ways that reduce both rendering quality and computational efficiency. To address these problems, our work introduces a rasterized approach to render the depth maps and surface normal maps of general 3D Gaussian splats. Our method not only significantly enhances shape reconstruction accuracy but also maintains the computational efficiency intrinsic to Gaussian Splatting. It achieves a Chamfer distance error comparable to NeuraLangelo on the DTU dataset and maintains similar computational efficiency as the original 3D GS methods. Our method is a significant advancement in Gaussian Splatting and can be directly integrated into existing Gaussian Splatting-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12900v5">PSAvatar: A Point-based Shape Model for Real-Time Head Avatar Animation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-24
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Despite much progress, achieving real-time high-fidelity head avatar animation is still difficult and existing methods have to trade-off between speed and quality. 3DMM based methods often fail to model non-facial structures such as eyeglasses and hairstyles, while neural implicit models suffer from deformation inflexibility and rendering inefficiency. Although 3D Gaussian has been demonstrated to possess promising capability for geometry representation and radiance field reconstruction, applying 3D Gaussian in head avatar creation remains a major challenge since it is difficult for 3D Gaussian to model the head shape variations caused by changing poses and expressions. In this paper, we introduce PSAvatar, a novel framework for animatable head avatar creation that utilizes discrete geometric primitive to create a parametric morphable shape model and employs 3D Gaussian for fine detail representation and high fidelity rendering. The parametric morphable shape model is a Point-based Morphable Shape Model (PMSM) which uses points instead of meshes for 3D representation to achieve enhanced representation flexibility. The PMSM first converts the FLAME mesh to points by sampling on the surfaces as well as off the meshes to enable the reconstruction of not only surface-like structures but also complex geometries such as eyeglasses and hairstyles. By aligning these points with the head shape in an analysis-by-synthesis manner, the PMSM makes it possible to utilize 3D Gaussian for fine detail representation and appearance modeling, thus enabling the creation of high-fidelity avatars. We show that PSAvatar can reconstruct high-fidelity head avatars of a variety of subjects and the avatars can be animated in real-time ($\ge$ 25 fps at a resolution of 512 $\times$ 512 ).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.14978v1">E2GS: Event Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-24
      | 💬 7pages,
    </div>
    <details class="paper-abstract">
      Event cameras, known for their high dynamic range, absence of motion blur, and low energy usage, have recently found a wide range of applications thanks to these attributes. In the past few years, the field of event-based 3D reconstruction saw remarkable progress, with the Neural Radiance Field (NeRF) based approach demonstrating photorealistic view synthesis results. However, the volume rendering paradigm of NeRF necessitates extensive training and rendering times. In this paper, we introduce Event Enhanced Gaussian Splatting (E2GS), a novel method that incorporates event data into Gaussian Splatting, which has recently made significant advances in the field of novel view synthesis. Our E2GS effectively utilizes both blurry images and event data, significantly improving image deblurring and producing high-quality novel view synthesis. Our comprehensive experiments on both synthetic and real-world datasets demonstrate our E2GS can generate visually appealing renderings while offering faster training and rendering speed (140 FPS). Our code is available at https://github.com/deguchihiroyuki/E2GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.01467v2">RaDe-GS: Rasterizing Depth in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-24
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has proven to be highly effective in novel view synthesis, achieving high-quality and real-time rendering. However, its potential for reconstructing detailed 3D shapes has not been fully explored. Existing methods often suffer from limited shape accuracy due to the discrete and unstructured nature of Gaussian splats, which complicates the shape extraction. While recent techniques like 2D GS have attempted to improve shape reconstruction, they often reformulate the Gaussian primitives in ways that reduce both rendering quality and computational efficiency. To address these problems, our work introduces a rasterized approach to render the depth maps and surface normal maps of general 3D Gaussian splats. Our method not only significantly enhances shape reconstruction accuracy but also maintains the computational efficiency intrinsic to Gaussian Splatting. It achieves a Chamfer distance error comparable to NeuraLangelo on the DTU dataset and maintains similar computational efficiency as the original 3D GS methods. Our method is a significant advancement in Gaussian Splatting and can be directly integrated into existing Gaussian Splatting-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16073v1">LGS: A Light-weight 4D Gaussian Splatting for Efficient Surgical Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-06-23
      | 💬 Accepted by MICCAI 2024. Project page: https://lgs-endo.github.io/
    </div>
    <details class="paper-abstract">
      The advent of 3D Gaussian Splatting (3D-GS) techniques and their dynamic scene modeling variants, 4D-GS, offers promising prospects for real-time rendering of dynamic surgical scenarios. However, the prerequisite for modeling dynamic scenes by a large number of Gaussian units, the high-dimensional Gaussian attributes and the high-resolution deformation fields, all lead to serve storage issues that hinder real-time rendering in resource-limited surgical equipment. To surmount these limitations, we introduce a Lightweight 4D Gaussian Splatting framework (LGS) that can liberate the efficiency bottlenecks of both rendering and storage for dynamic endoscopic reconstruction. Specifically, to minimize the redundancy of Gaussian quantities, we propose Deformation-Aware Pruning by gauging the impact of each Gaussian on deformation. Concurrently, to reduce the redundancy of Gaussian attributes, we simplify the representation of textures and lighting in non-crucial areas by pruning the dimensions of Gaussian attributes. We further resolve the feature field redundancy caused by the high resolution of 4D neural spatiotemporal encoder for modeling dynamic scenes via a 4D feature field condensation. Experiments on public benchmarks demonstrate efficacy of LGS in terms of a compression rate exceeding 9 times while maintaining the pleasing visual quality and real-time rendering efficiency. LGS confirms a substantial step towards its application in robotic surgical services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11836v2">RetinaGS: Scalable Training for Dense Scene Rendering with Billion-Scale 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-06-22
    </div>
    <details class="paper-abstract">
      In this work, we explore the possibility of training high-parameter 3D Gaussian splatting (3DGS) models on large-scale, high-resolution datasets. We design a general model parallel training method for 3DGS, named RetinaGS, which uses a proper rendering equation and can be applied to any scene and arbitrary distribution of Gaussian primitives. It enables us to explore the scaling behavior of 3DGS in terms of primitive numbers and training resolutions that were difficult to explore before and surpass previous state-of-the-art reconstruction quality. We observe a clear positive trend of increasing visual quality when increasing primitive numbers with our method. We also demonstrate the first attempt at training a 3DGS model with more than one billion primitives on the full MatrixCity dataset that attains a promising visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12477v3">Gaussian Control with Hierarchical Semantic Graphs in 3D Human Recovery</a></div>
    <div class="paper-meta">
      📅 2024-06-22
    </div>
    <details class="paper-abstract">
      Although 3D Gaussian Splatting (3DGS) has recently made progress in 3D human reconstruction, it primarily relies on 2D pixel-level supervision, overlooking the geometric complexity and topological relationships of different body parts. To address this gap, we introduce the Hierarchical Graph Human Gaussian Control (HUGS) framework for achieving high-fidelity 3D human reconstruction. Our approach involves leveraging explicitly semantic priors of body parts to ensure the consistency of geometric topology, thereby enabling the capture of the complex geometrical and topological associations among body parts. Additionally, we disentangle high-frequency features from global human features to refine surface details in body parts. Extensive experiments demonstrate that our method exhibits superior performance in human body reconstruction, particularly in enhancing surface details and accurately reconstructing body part junctions. Codes are available at https://wanghongsheng01.github.io/HUGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12806v3">MOSS: Motion-based 3D Clothed Human Synthesis from Monocular Video</a></div>
    <div class="paper-meta">
      📅 2024-06-22
      | 💬 arXiv admin note: text overlap with arXiv:1710.03746 by other authors
    </div>
    <details class="paper-abstract">
      Single-view clothed human reconstruction holds a central position in virtual reality applications, especially in contexts involving intricate human motions. It presents notable challenges in achieving realistic clothing deformation. Current methodologies often overlook the influence of motion on surface deformation, resulting in surfaces lacking the constraints imposed by global motion. To overcome these limitations, we introduce an innovative framework, Motion-Based 3D Clo}thed Humans Synthesis (MOSS), which employs kinematic information to achieve motion-aware Gaussian split on the human surface. Our framework consists of two modules: Kinematic Gaussian Locating Splatting (KGAS) and Surface Deformation Detector (UID). KGAS incorporates matrix-Fisher distribution to propagate global motion across the body surface. The density and rotation factors of this distribution explicitly control the Gaussians, thereby enhancing the realism of the reconstructed surface. Additionally, to address local occlusions in single-view, based on KGAS, UID identifies significant surfaces, and geometric reconstruction is performed to compensate for these deformations. Experimental results demonstrate that MOSS achieves state-of-the-art visual quality in 3D clothed human synthesis from monocular videos. Notably, we improve the Human NeRF and the Gaussian Splatting by 33.94% and 16.75% in LPIPS* respectively. Codes are available at https://wanghongsheng01.github.io/MOSS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15643v1">Taming 3DGS: High-Quality Radiance Fields with Limited Resources</a></div>
    <div class="paper-meta">
      📅 2024-06-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has transformed novel-view synthesis with its fast, interpretable, and high-fidelity rendering. However, its resource requirements limit its usability. Especially on constrained devices, training performance degrades quickly and often cannot complete due to excessive memory consumption of the model. The method converges with an indefinite number of Gaussians -- many of them redundant -- making rendering unnecessarily slow and preventing its usage in downstream tasks that expect fixed-size inputs. To address these issues, we tackle the challenges of training and rendering 3DGS models on a budget. We use a guided, purely constructive densification process that steers densification toward Gaussians that raise the reconstruction quality. Model size continuously increases in a controlled manner towards an exact budget, using score-based densification of Gaussians with training-time priors that measure their contribution. We further address training speed obstacles: following a careful analysis of 3DGS' original pipeline, we derive faster, numerically equivalent solutions for gradient computation and attribute updates, including an alternative parallelization for efficient backpropagation. We also propose quality-preserving approximations where suitable to reduce training time even further. Taken together, these enhancements yield a robust, scalable solution with reduced training times, lower compute and memory requirements, and high quality. Our evaluation shows that in a budgeted setting, we obtain competitive quality metrics with 3DGS while achieving a 4--5x reduction in both model size and training time. With more generous budgets, our measured quality surpasses theirs. These advances open the door for novel-view synthesis in constrained environments, e.g., mobile devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14978v1">E2GS: Event Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-21
      | 💬 7pages,
    </div>
    <details class="paper-abstract">
      Event cameras, known for their high dynamic range, absence of motion blur, and low energy usage, have recently found a wide range of applications thanks to these attributes. In the past few years, the field of event-based 3D reconstruction saw remarkable progress, with the Neural Radiance Field (NeRF) based approach demonstrating photorealistic view synthesis results. However, the volume rendering paradigm of NeRF necessitates extensive training and rendering times. In this paper, we introduce Event Enhanced Gaussian Splatting (E2GS), a novel method that incorporates event data into Gaussian Splatting, which has recently made significant advances in the field of novel view synthesis. Our E2GS effectively utilizes both blurry images and event data, significantly improving image deblurring and producing high-quality novel view synthesis. Our comprehensive experiments on both synthetic and real-world datasets demonstrate our E2GS can generate visually appealing renderings while offering faster training and rendering speed (140 FPS). Our code is available at https://github.com/deguchihiroyuki/E2GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.14978v1">E2GS: Event Enhanced Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-21
      | 💬 7pages,
    </div>
    <details class="paper-abstract">
      Event cameras, known for their high dynamic range, absence of motion blur, and low energy usage, have recently found a wide range of applications thanks to these attributes. In the past few years, the field of event-based 3D reconstruction saw remarkable progress, with the Neural Radiance Field (NeRF) based approach demonstrating photorealistic view synthesis results. However, the volume rendering paradigm of NeRF necessitates extensive training and rendering times. In this paper, we introduce Event Enhanced Gaussian Splatting (E2GS), a novel method that incorporates event data into Gaussian Splatting, which has recently made significant advances in the field of novel view synthesis. Our E2GS effectively utilizes both blurry images and event data, significantly improving image deblurring and producing high-quality novel view synthesis. Our comprehensive experiments on both synthetic and real-world datasets demonstrate our E2GS can generate visually appealing renderings while offering faster training and rendering speed (140 FPS). Our code is available at https://github.com/deguchihiroyuki/E2GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.10318v2">SRGS: Super-Resolution 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-19
      | 💬 The first to focus on the HRNVS of 3DGS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has gained popularity as a novel explicit 3D representation. This approach relies on the representation power of Gaussian primitives to provide a high-quality rendering. However, primitives optimized at low resolution inevitably exhibit sparsity and texture deficiency, posing a challenge for achieving high-resolution novel view synthesis (HRNVS). To address this problem, we propose Super-Resolution 3D Gaussian Splatting (SRGS) to perform the optimization in a high-resolution (HR) space. The sub-pixel constraint is introduced for the increased viewpoints in HR space, exploiting the sub-pixel cross-view information of the multiple low-resolution (LR) views. The gradient accumulated from more viewpoints will facilitate the densification of primitives. Furthermore, a pre-trained 2D super-resolution model is integrated with the sub-pixel constraint, enabling these dense primitives to learn faithful texture features. In general, our method focuses on densification and texture learning to effectively enhance the representation ability of primitives. Experimentally, our method achieves high rendering quality on HRNVS only with LR inputs, outperforming state-of-the-art methods on challenging datasets such as Mip-NeRF 360 and Tanks & Temples. Related codes will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13099v1">Sampling 3D Gaussian Scenes in Seconds with Latent Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-06-18
    </div>
    <details class="paper-abstract">
      We present a latent diffusion model over 3D scenes, that can be trained using only 2D image data. To achieve this, we first design an autoencoder that maps multi-view images to 3D Gaussian splats, and simultaneously builds a compressed latent representation of these splats. Then, we train a multi-view diffusion model over the latent space to learn an efficient generative model. This pipeline does not require object masks nor depths, and is suitable for complex scenes with arbitrary camera positions. We conduct careful experiments on two large-scale datasets of complex real-world scenes -- MVImgNet and RealEstate10K. We show that our approach enables generating 3D scenes in as little as 0.2 seconds, either from scratch, from a single input view, or from sparse input views. It produces diverse and high-quality results while running an order of magnitude faster than non-latent diffusion models and earlier NeRF-based generative models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10318v2">SRGS: Super-Resolution 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-18
      | 💬 The first to focus on the HRNVS of 3DGS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has gained popularity as a novel explicit 3D representation. This approach relies on the representation power of Gaussian primitives to provide a high-quality rendering. However, primitives optimized at low resolution inevitably exhibit sparsity and texture deficiency, posing a challenge for achieving high-resolution novel view synthesis (HRNVS). To address this problem, we propose Super-Resolution 3D Gaussian Splatting (SRGS) to perform the optimization in a high-resolution (HR) space. The sub-pixel constraint is introduced for the increased viewpoints in HR space, exploiting the sub-pixel cross-view information of the multiple low-resolution (LR) views. The gradient accumulated from more viewpoints will facilitate the densification of primitives. Furthermore, a pre-trained 2D super-resolution model is integrated with the sub-pixel constraint, enabling these dense primitives to learn faithful texture features. In general, our method focuses on densification and texture learning to effectively enhance the representation ability of primitives. Experimentally, our method achieves high rendering quality on HRNVS only with LR inputs, outperforming state-of-the-art methods on challenging datasets such as Mip-NeRF 360 and Tanks & Temples. Related codes will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12155v2">Embracing Radiance Field Rendering in 6G: Over-the-Air Training and Inference with 3D Contents</a></div>
    <div class="paper-meta">
      📅 2024-06-18
      | 💬 16 pages,7 figures
    </div>
    <details class="paper-abstract">
      The efficient representation, transmission, and reconstruction of three-dimensional (3D) contents are becoming increasingly important for sixth-generation (6G) networks that aim to merge virtual and physical worlds for offering immersive communication experiences. Neural radiance field (NeRF) and 3D Gaussian splatting (3D-GS) have recently emerged as two promising 3D representation techniques based on radiance field rendering, which are able to provide photorealistic rendering results for complex scenes. Therefore, embracing NeRF and 3D-GS in 6G networks is envisioned to be a prominent solution to support emerging 3D applications with enhanced quality of experience. This paper provides a comprehensive overview on the integration of NeRF and 3D-GS in 6G. First, we review the basics of the radiance field rendering techniques, and highlight their applications and implementation challenges over wireless networks. Next, we consider the over-the-air training of NeRF and 3D-GS models over wireless networks by presenting various learning techniques. We particularly focus on the federated learning design over a hierarchical device-edge-cloud architecture, which is suitable for exploiting distributed data and computing resources over 6G networks to train large models representing large-scale scenes. Then, we consider the over-the-air rendering of NeRF and 3D-GS models at wireless network edge. We present three practical rendering architectures, namely local, remote, and co-rendering, respectively, and provide model compression approaches to facilitate the transmission of radiance field models for rendering. We also present rendering acceleration approaches and joint computation and communication designs to enhance the rendering efficiency. In a case study, we propose a new semantic communication enabled 3D content transmission design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.10318v2">SRGS: Super-Resolution 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-18
      | 💬 The first to focus on the HRNVS of 3DGS
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has gained popularity as a novel explicit 3D representation. This approach relies on the representation power of Gaussian primitives to provide a high-quality rendering. However, primitives optimized at low resolution inevitably exhibit sparsity and texture deficiency, posing a challenge for achieving high-resolution novel view synthesis (HRNVS). To address this problem, we propose Super-Resolution 3D Gaussian Splatting (SRGS) to perform the optimization in a high-resolution (HR) space. The sub-pixel constraint is introduced for the increased viewpoints in HR space, exploiting the sub-pixel cross-view information of the multiple low-resolution (LR) views. The gradient accumulated from more viewpoints will facilitate the densification of primitives. Furthermore, a pre-trained 2D super-resolution model is integrated with the sub-pixel constraint, enabling these dense primitives to learn faithful texture features. In general, our method focuses on densification and texture learning to effectively enhance the representation ability of primitives. Experimentally, our method achieves high rendering quality on HRNVS only with LR inputs, outperforming state-of-the-art methods on challenging datasets such as Mip-NeRF 360 and Tanks & Temples. Related codes will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12080v1">A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 Project Page: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
    </div>
    <details class="paper-abstract">
      Novel view synthesis has seen major advances in recent years, with 3D Gaussian splatting offering an excellent level of visual quality, fast training and real-time rendering. However, the resources needed for training and rendering inevitably limit the size of the captured scenes that can be represented with good visual quality. We introduce a hierarchy of 3D Gaussians that preserves visual quality for very large scenes, while offering an efficient Level-of-Detail (LOD) solution for efficient rendering of distant content with effective level selection and smooth transitions between levels.We introduce a divide-and-conquer approach that allows us to train very large scenes in independent chunks. We consolidate the chunks into a hierarchy that can be optimized to further improve visual quality of Gaussians merged into intermediate nodes. Very large captures typically have sparse coverage of the scene, presenting many challenges to the original 3D Gaussian splatting training method; we adapt and regularize training to account for these issues. We present a complete solution, that enables real-time rendering of very large scenes and can adapt to available resources thanks to our LOD method. We show results for captured scenes with up to tens of thousands of images with a simple and affordable rig, covering trajectories of up to several kilometers and lasting up to one hour. Project Page: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10625v2">Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 CVPRW
    </div>
    <details class="paper-abstract">
      NeRF-based 3D-aware Generative Adversarial Networks (GANs) like EG3D or GIRAFFE have shown very high rendering quality under large representational variety. However, rendering with Neural Radiance Fields poses challenges for 3D applications: First, the significant computational demands of NeRF rendering preclude its use on low-power devices, such as mobiles and VR/AR headsets. Second, implicit representations based on neural networks are difficult to incorporate into explicit 3D scenes, such as VR environments or video games. 3D Gaussian Splatting (3DGS) overcomes these limitations by providing an explicit 3D representation that can be rendered efficiently at high frame rates. In this work, we present a novel approach that combines the high rendering quality of NeRF-based 3D-aware GANs with the flexibility and computational advantages of 3DGS. By training a decoder that maps implicit NeRF representations to explicit 3D Gaussian Splatting attributes, we can integrate the representational diversity and quality of 3D GANs into the ecosystem of 3D Gaussian Splatting for the first time. Additionally, our approach allows for a high resolution GAN inversion and real-time GAN editing with 3D Gaussian Splatting scenes. Project page: florian-barthel.github.io/gaussian_decoder
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09591v2">3D Gaussian Splatting as Markov Chain Monte Carlo</a></div>
    <div class="paper-meta">
      📅 2024-06-17
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting has recently become popular for neural rendering, current methods rely on carefully engineered cloning and splitting strategies for placing Gaussians, which can lead to poor-quality renderings, and reliance on a good initialization. In this work, we rethink the set of 3D Gaussians as a random sample drawn from an underlying probability distribution describing the physical representation of the scene-in other words, Markov Chain Monte Carlo (MCMC) samples. Under this view, we show that the 3D Gaussian updates can be converted as Stochastic Gradient Langevin Dynamics (SGLD) updates by simply introducing noise. We then rewrite the densification and pruning strategies in 3D Gaussian Splatting as simply a deterministic state transition of MCMC samples, removing these heuristics from the framework. To do so, we revise the 'cloning' of Gaussians into a relocalization scheme that approximately preserves sample probability. To encourage efficient use of Gaussians, we introduce a regularizer that promotes the removal of unused Gaussians. On various standard evaluation scenes, we show that our method provides improved rendering quality, easy control over the number of Gaussians, and robustness to initialization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.10625v2">Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks</a></div>
    <div class="paper-meta">
      📅 2024-06-17
      | 💬 CVPRW
    </div>
    <details class="paper-abstract">
      NeRF-based 3D-aware Generative Adversarial Networks (GANs) like EG3D or GIRAFFE have shown very high rendering quality under large representational variety. However, rendering with Neural Radiance Fields poses challenges for 3D applications: First, the significant computational demands of NeRF rendering preclude its use on low-power devices, such as mobiles and VR/AR headsets. Second, implicit representations based on neural networks are difficult to incorporate into explicit 3D scenes, such as VR environments or video games. 3D Gaussian Splatting (3DGS) overcomes these limitations by providing an explicit 3D representation that can be rendered efficiently at high frame rates. In this work, we present a novel approach that combines the high rendering quality of NeRF-based 3D-aware GANs with the flexibility and computational advantages of 3DGS. By training a decoder that maps implicit NeRF representations to explicit 3D Gaussian Splatting attributes, we can integrate the representational diversity and quality of 3D GANs into the ecosystem of 3D Gaussian Splatting for the first time. Additionally, our approach allows for a high resolution GAN inversion and real-time GAN editing with 3D Gaussian Splatting scenes. Project page: florian-barthel.github.io/gaussian_decoder
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00451v2">FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-16
      | 💬 Project page: https://zehaozhu.github.io/FSGS/
    </div>
    <details class="paper-abstract">
      Novel view synthesis from limited observations remains an important and persistent task. However, high efficiency in existing NeRF-based few-shot view synthesis is often compromised to obtain an accurate 3D representation. To address this challenge, we propose a few-shot view synthesis framework based on 3D Gaussian Splatting that enables real-time and photo-realistic view synthesis with as few as three training views. The proposed method, dubbed FSGS, handles the extremely sparse initialized SfM points with a thoughtfully designed Gaussian Unpooling process. Our method iteratively distributes new Gaussians around the most representative locations, subsequently infilling local details in vacant areas. We also integrate a large-scale pre-trained monocular depth estimator within the Gaussians optimization process, leveraging online augmented views to guide the geometric optimization towards an optimal solution. Starting from sparse points observed from limited input viewpoints, our FSGS can accurately grow into unseen regions, comprehensively covering the scene and boosting the rendering quality of novel views. Overall, FSGS achieves state-of-the-art performance in both accuracy and rendering efficiency across diverse datasets, including LLFF, Mip-NeRF360, and Blender. Project website: https://zehaozhu.github.io/FSGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10788v1">Physically Embodied Gaussian Splatting: A Realtime Correctable World Model for Robotics</a></div>
    <div class="paper-meta">
      📅 2024-06-16
    </div>
    <details class="paper-abstract">
      For robots to robustly understand and interact with the physical world, it is highly beneficial to have a comprehensive representation - modelling geometry, physics, and visual observations - that informs perception, planning, and control algorithms. We propose a novel dual Gaussian-Particle representation that models the physical world while (i) enabling predictive simulation of future states and (ii) allowing online correction from visual observations in a dynamic world. Our representation comprises particles that capture the geometrical aspect of objects in the world and can be used alongside a particle-based physics system to anticipate physically plausible future states. Attached to these particles are 3D Gaussians that render images from any viewpoint through a splatting process thus capturing the visual state. By comparing the predicted and observed images, our approach generates visual forces that correct the particle positions while respecting known physical constraints. By integrating predictive physical modelling with continuous visually-derived corrections, our unified representation reasons about the present and future while synchronizing with reality. Our system runs in realtime at 30Hz using only 3 cameras. We validate our approach on 2D and 3D tracking tasks as well as photometric reconstruction quality. Videos are found at https://embodied-gaussians.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10373v1">Wild-GS: Real-Time Novel View Synthesis from Unconstrained Photo Collections</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Photographs captured in unstructured tourist environments frequently exhibit variable appearances and transient occlusions, challenging accurate scene reconstruction and inducing artifacts in novel view synthesis. Although prior approaches have integrated the Neural Radiance Field (NeRF) with additional learnable modules to handle the dynamic appearances and eliminate transient objects, their extensive training demands and slow rendering speeds limit practical deployments. Recently, 3D Gaussian Splatting (3DGS) has emerged as a promising alternative to NeRF, offering superior training and inference efficiency along with better rendering quality. This paper presents Wild-GS, an innovative adaptation of 3DGS optimized for unconstrained photo collections while preserving its efficiency benefits. Wild-GS determines the appearance of each 3D Gaussian by their inherent material attributes, global illumination and camera properties per image, and point-level local variance of reflectance. Unlike previous methods that model reference features in image space, Wild-GS explicitly aligns the pixel appearance features to the corresponding local Gaussians by sampling the triplane extracted from the reference image. This novel design effectively transfers the high-frequency detailed appearance of the reference view to 3D space and significantly expedites the training process. Furthermore, 2D visibility maps and depth regularization are leveraged to mitigate the transient effects and constrain the geometry, respectively. Extensive experiments demonstrate that Wild-GS achieves state-of-the-art rendering performance and the highest efficiency in both training and inference among all the existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10324v1">L4GM: Large 4D Gaussian Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 Project page: https://research.nvidia.com/labs/toronto-ai/l4gm
    </div>
    <details class="paper-abstract">
      We present L4GM, the first 4D Large Reconstruction Model that produces animated objects from a single-view video input -- in a single feed-forward pass that takes only a second. Key to our success is a novel dataset of multiview videos containing curated, rendered animated objects from Objaverse. This dataset depicts 44K diverse objects with 110K animations rendered in 48 viewpoints, resulting in 12M videos with a total of 300M frames. We keep our L4GM simple for scalability and build directly on top of LGM, a pretrained 3D Large Reconstruction Model that outputs 3D Gaussian ellipsoids from multiview image input. L4GM outputs a per-frame 3D Gaussian Splatting representation from video frames sampled at a low fps and then upsamples the representation to a higher fps to achieve temporal smoothness. We add temporal self-attention layers to the base LGM to help it learn consistency across time, and utilize a per-timestep multiview rendering loss to train the model. The representation is upsampled to a higher framerate by training an interpolation model which produces intermediate 3D Gaussian representations. We showcase that L4GM that is only trained on synthetic data generalizes extremely well on in-the-wild videos, producing high quality animated 3D assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10111v1">GaussianSR: 3D Gaussian Super-Resolution with 2D Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2024-06-14
    </div>
    <details class="paper-abstract">
      Achieving high-resolution novel view synthesis (HRNVS) from low-resolution input views is a challenging task due to the lack of high-resolution data. Previous methods optimize high-resolution Neural Radiance Field (NeRF) from low-resolution input views but suffer from slow rendering speed. In this work, we base our method on 3D Gaussian Splatting (3DGS) due to its capability of producing high-quality images at a faster rendering speed. To alleviate the shortage of data for higher-resolution synthesis, we propose to leverage off-the-shelf 2D diffusion priors by distilling the 2D knowledge into 3D with Score Distillation Sampling (SDS). Nevertheless, applying SDS directly to Gaussian-based 3D super-resolution leads to undesirable and redundant 3D Gaussian primitives, due to the randomness brought by generative priors. To mitigate this issue, we introduce two simple yet effective techniques to reduce stochastic disturbances introduced by SDS. Specifically, we 1) shrink the range of diffusion timestep in SDS with an annealing strategy; 2) randomly discard redundant Gaussian primitives during densification. Extensive experiments have demonstrated that our proposed GaussainSR can attain high-quality results for HRNVS with only low-resolution inputs on both synthetic and real-world datasets. Project page: https://chchnii.github.io/GaussianSR/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09850v1">GradeADreamer: Enhanced Text-to-3D Generation Using Gaussian Splatting and Multi-View Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-06-14
      | 💬 Code: https://github.com/trapoom555/GradeADreamer
    </div>
    <details class="paper-abstract">
      Text-to-3D generation has shown promising results, yet common challenges such as the Multi-face Janus problem and extended generation time for high-quality assets. In this paper, we address these issues by introducing a novel three-stage training pipeline called GradeADreamer. This pipeline is capable of producing high-quality assets with a total generation time of under 30 minutes using only a single RTX 3090 GPU. Our proposed method employs a Multi-view Diffusion Model, MVDream, to generate Gaussian Splats as a prior, followed by refining geometry and texture using StableDiffusion. Experimental results demonstrate that our approach significantly mitigates the Multi-face Janus problem and achieves the highest average user preference ranking compared to previous state-of-the-art methods. The project code is available at https://github.com/trapoom555/GradeADreamer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08920v2">AV-GS: Learning Material and Geometry Aware Priors for Novel View Acoustic Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-06-14
    </div>
    <details class="paper-abstract">
      Novel view acoustic synthesis (NVAS) aims to render binaural audio at any target viewpoint, given a mono audio emitted by a sound source at a 3D scene. Existing methods have proposed NeRF-based implicit models to exploit visual cues as a condition for synthesizing binaural audio. However, in addition to low efficiency originating from heavy NeRF rendering, these methods all have a limited ability of characterizing the entire scene environment such as room geometry, material properties, and the spatial relation between the listener and sound source. To address these issues, we propose a novel Audio-Visual Gaussian Splatting (AV-GS) model. To obtain a material-aware and geometry-aware condition for audio synthesis, we learn an explicit point-based scene representation with an audio-guidance parameter on locally initialized Gaussian points, taking into account the space relation from the listener and sound source. To make the visual scene model audio adaptive, we propose a point densification and pruning strategy to optimally distribute the Gaussian points, with the per-point contribution in sound propagation (e.g., more points needed for texture-less wall surfaces as they affect sound path diversion). Extensive experiments validate the superiority of our AV-GS over existing alternatives on the real-world RWAS and simulation-based SoundSpaces datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02720v2">3D-HGS: 3D Half-Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-13
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Photo-realistic 3D Reconstruction is a fundamental problem in 3D computer vision. This domain has seen considerable advancements owing to the advent of recent neural rendering techniques. These techniques predominantly aim to focus on learning volumetric representations of 3D scenes and refining these representations via loss functions derived from rendering. Among these, 3D Gaussian Splatting (3D-GS) has emerged as a significant method, surpassing Neural Radiance Fields (NeRFs). 3D-GS uses parameterized 3D Gaussians for modeling both spatial locations and color information, combined with a tile-based fast rendering technique. Despite its superior rendering performance and speed, the use of 3D Gaussian kernels has inherent limitations in accurately representing discontinuous functions, notably at edges and corners for shape discontinuities, and across varying textures for color discontinuities. To address this problem, we propose to employ 3D Half-Gaussian (3D-HGS) kernels, which can be used as a plug-and-play kernel. Our experiments demonstrate their capability to improve the performance of current 3D-GS related methods and achieve state-of-the-art rendering performance on various datasets without compromising rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09395v1">Modeling Ambient Scene Dynamics for Free-view Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      We introduce a novel method for dynamic free-view synthesis of an ambient scenes from a monocular capture bringing a immersive quality to the viewing experience. Our method builds upon the recent advancements in 3D Gaussian Splatting (3DGS) that can faithfully reconstruct complex static scenes. Previous attempts to extend 3DGS to represent dynamics have been confined to bounded scenes or require multi-camera captures, and often fail to generalize to unseen motions, limiting their practical application. Our approach overcomes these constraints by leveraging the periodicity of ambient motions to learn the motion trajectory model, coupled with careful regularization. We also propose important practical strategies to improve the visual quality of the baseline 3DGS static reconstructions and to improve memory efficiency critical for GPU-memory intensive learning. We demonstrate high-quality photorealistic novel view synthesis of several ambient natural scenes with intricate textures and fine structural elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04251v2">Gaussian Splatting with Localized Points Management</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Point management is a critical component in optimizing 3D Gaussian Splatting (3DGS) models, as the point initiation (e.g., via structure from motion) is distributionally inappropriate. Typically, the Adaptive Density Control (ADC) algorithm is applied, leveraging view-averaged gradient magnitude thresholding for point densification, opacity thresholding for pruning, and regular all-points opacity reset. However, we reveal that this strategy is limited in tackling intricate/special image regions (e.g., transparent) as it is unable to identify all the 3D zones that require point densification, and lacking an appropriate mechanism to handle the ill-conditioned points with negative impacts (occlusion due to false high opacity). To address these limitations, we propose a Localized Point Management (LPM) strategy, capable of identifying those error-contributing zones in the highest demand for both point addition and geometry calibration. Zone identification is achieved by leveraging the underlying multiview geometry constraints, with the guidance of image rendering errors. We apply point densification in the identified zone, whilst resetting the opacity of those points residing in front of these regions so that a new opportunity is created to correct ill-conditioned points. Serving as a versatile plugin, LPM can be seamlessly integrated into existing 3D Gaussian Splatting models. Experimental evaluation across both static 3D and dynamic 4D scenes validate the efficacy of our LPM strategy in boosting a variety of existing 3DGS models both quantitatively and qualitatively. Notably, LPM improves both vanilla 3DGS and SpaceTimeGS to achieve state-of-the-art rendering quality while retaining real-time speeds, outperforming on challenging datasets such as Tanks & Temples and the Neural 3D Video Dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2312.13729v5">Gaussian Splatting with NeRF-based Color and Opacity</a></div>
    <div class="paper-meta">
      📅 2024-06-13
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRFs) have demonstrated the remarkable potential of neural networks to capture the intricacies of 3D objects. By encoding the shape and color information within neural network weights, NeRFs excel at producing strikingly sharp novel views of 3D objects. Recently, numerous generalizations of NeRFs utilizing generative models have emerged, expanding its versatility. In contrast, Gaussian Splatting (GS) offers a similar render quality with faster training and inference as it does not need neural networks to work. It encodes information about the 3D objects in the set of Gaussian distributions that can be rendered in 3D similarly to classical meshes. Unfortunately, GS are difficult to condition since they usually require circa hundred thousand Gaussian components. To mitigate the caveats of both models, we propose a hybrid model Viewing Direction Gaussian Splatting (VDGS) that uses GS representation of the 3D object's shape and NeRF-based encoding of color and opacity. Our model uses Gaussian distributions with trainable positions (i.e. means of Gaussian), shape (i.e. covariance of Gaussian), color and opacity, and a neural network that takes Gaussian parameters and viewing direction to produce changes in the said color and opacity. As a result, our model better describes shadows, light reflections, and the transparency of 3D objects without adding additional texture and light components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08488v1">ICE-G: Image Conditional Editing of 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-06-12
      | 💬 Accepted to CVPR AI4CC Workshop 2024. Project page: https://ice-gaussian.github.io
    </div>
    <details class="paper-abstract">
      Recently many techniques have emerged to create high quality 3D assets and scenes. When it comes to editing of these objects, however, existing approaches are either slow, compromise on quality, or do not provide enough customization. We introduce a novel approach to quickly edit a 3D model from a single reference view. Our technique first segments the edit image, and then matches semantically corresponding regions across chosen segmented dataset views using DINO features. A color or texture change from a particular region of the edit image can then be applied to other views automatically in a semantically sensible manner. These edited views act as an updated dataset to further train and re-style the 3D scene. The end-result is therefore an edited 3D model. Our framework enables a wide variety of editing tasks such as manual local edits, correspondence based style transfer from any example image, and a combination of different styles from multiple example images. We use Gaussian Splats as our primary 3D representation due to their speed and ease of local editing, but our technique works for other methods such as NeRFs as well. We show through multiple examples that our method produces higher quality results while offering fine-grained control of editing. Project page: ice-gaussian.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08300v1">From Chaos to Clarity: 3DGS in the Dark</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Novel view synthesis from raw images provides superior high dynamic range (HDR) information compared to reconstructions from low dynamic range RGB images. However, the inherent noise in unprocessed raw images compromises the accuracy of 3D scene representation. Our study reveals that 3D Gaussian Splatting (3DGS) is particularly susceptible to this noise, leading to numerous elongated Gaussian shapes that overfit the noise, thereby significantly degrading reconstruction quality and reducing inference speed, especially in scenarios with limited views. To address these issues, we introduce a novel self-supervised learning framework designed to reconstruct HDR 3DGS from a limited number of noisy raw images. This framework enhances 3DGS by integrating a noise extractor and employing a noise-robust reconstruction loss that leverages a noise distribution prior. Experimental results show that our method outperforms LDR/HDR 3DGS and previous state-of-the-art (SOTA) self-supervised and supervised pre-trained models in both reconstruction quality and inference speed on the RawNeRF dataset across a broad range of training views. Code can be found in \url{https://lizhihao6.github.io/Raw3DGS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.13729v5">Gaussian Splatting with NeRF-based Color and Opacity</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRFs) have demonstrated the remarkable potential of neural networks to capture the intricacies of 3D objects. By encoding the shape and color information within neural network weights, NeRFs excel at producing strikingly sharp novel views of 3D objects. Recently, numerous generalizations of NeRFs utilizing generative models have emerged, expanding its versatility. In contrast, Gaussian Splatting (GS) offers a similar render quality with faster training and inference as it does not need neural networks to work. It encodes information about the 3D objects in the set of Gaussian distributions that can be rendered in 3D similarly to classical meshes. Unfortunately, GS are difficult to condition since they usually require circa hundred thousand Gaussian components. To mitigate the caveats of both models, we propose a hybrid model Viewing Direction Gaussian Splatting (VDGS) that uses GS representation of the 3D object's shape and NeRF-based encoding of color and opacity. Our model uses Gaussian distributions with trainable positions (i.e. means of Gaussian), shape (i.e. covariance of Gaussian), color and opacity, and a neural network that takes Gaussian parameters and viewing direction to produce changes in the said color and opacity. As a result, our model better describes shadows, light reflections, and the transparency of 3D objects without adding additional texture and light components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.13729v5">Gaussian Splatting with NeRF-based Color and Opacity</a></div>
    <div class="paper-meta">
      📅 2024-06-12
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRFs) have demonstrated the remarkable potential of neural networks to capture the intricacies of 3D objects. By encoding the shape and color information within neural network weights, NeRFs excel at producing strikingly sharp novel views of 3D objects. Recently, numerous generalizations of NeRFs utilizing generative models have emerged, expanding its versatility. In contrast, Gaussian Splatting (GS) offers a similar render quality with faster training and inference as it does not need neural networks to work. It encodes information about the 3D objects in the set of Gaussian distributions that can be rendered in 3D similarly to classical meshes. Unfortunately, GS are difficult to condition since they usually require circa hundred thousand Gaussian components. To mitigate the caveats of both models, we propose a hybrid model Viewing Direction Gaussian Splatting (VDGS) that uses GS representation of the 3D object's shape and NeRF-based encoding of color and opacity. Our model uses Gaussian distributions with trainable positions (i.e. means of Gaussian), shape (i.e. covariance of Gaussian), color and opacity, and a neural network that takes Gaussian parameters and viewing direction to produce changes in the said color and opacity. As a result, our model better describes shadows, light reflections, and the transparency of 3D objects without adding additional texture and light components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07499v1">Trim 3D Gaussian Splatting for Accurate Geometry Representation</a></div>
    <div class="paper-meta">
      📅 2024-06-11
      | 💬 Project page: https://trimgs.github.io/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Trim 3D Gaussian Splatting (TrimGS) to reconstruct accurate 3D geometry from images. Previous arts for geometry reconstruction from 3D Gaussians mainly focus on exploring strong geometry regularization. Instead, from a fresh perspective, we propose to obtain accurate 3D geometry of a scene by Gaussian trimming, which selectively removes the inaccurate geometry while preserving accurate structures. To achieve this, we analyze the contributions of individual 3D Gaussians and propose a contribution-based trimming strategy to remove the redundant or inaccurate Gaussians. Furthermore, our experimental and theoretical analyses reveal that a relatively small Gaussian scale is a non-negligible factor in representing and optimizing the intricate details. Therefore the proposed TrimGS maintains relatively small Gaussian scales. In addition, TrimGS is also compatible with the effective geometry regularization strategies in previous arts. When combined with the original 3DGS and the state-of-the-art 2DGS, TrimGS consistently yields more accurate geometry and higher perceptual quality. Our project page is https://trimgs.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07207v2">GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-11
    </div>
    <details class="paper-abstract">
      We present GALA3D, generative 3D GAussians with LAyout-guided control, for effective compositional text-to-3D generation. We first utilize large language models (LLMs) to generate the initial layout and introduce a layout-guided 3D Gaussian representation for 3D content generation with adaptive geometric constraints. We then propose an instance-scene compositional optimization mechanism with conditioned diffusion to collaboratively generate realistic 3D scenes with consistent geometry, texture, scale, and accurate interactions among multiple objects while simultaneously adjusting the coarse layout priors extracted from the LLMs to align with the generated scene. Experiments show that GALA3D is a user-friendly, end-to-end framework for state-of-the-art scene-level 3D content generation and controllable editing while ensuring the high fidelity of object-level entities within the scene. The source codes and models will be available at gala3d.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03659v2">A Construct-Optimize Approach to Sparse View Synthesis without Camera Pose</a></div>
    <div class="paper-meta">
      📅 2024-06-10
    </div>
    <details class="paper-abstract">
      Novel view synthesis from a sparse set of input images is a challenging problem of great practical interest, especially when camera poses are absent or inaccurate. Direct optimization of camera poses and usage of estimated depths in neural radiance field algorithms usually do not produce good results because of the coupling between poses and depths, and inaccuracies in monocular depth estimation. In this paper, we leverage the recent 3D Gaussian splatting method to develop a novel construct-and-optimize method for sparse view synthesis without camera poses. Specifically, we construct a solution progressively by using monocular depth and projecting pixels back into the 3D world. During construction, we optimize the solution by detecting 2D correspondences between training views and the corresponding rendered images. We develop a unified differentiable pipeline for camera registration and adjustment of both camera poses and depths, followed by back-projection. We also introduce a novel notion of an expected surface in Gaussian splatting, which is critical to our optimization. These steps enable a coarse solution, which can then be low-pass filtered and refined using standard optimization methods. We demonstrate results on the Tanks and Temples and Static Hikes datasets with as few as three widely-spaced views, showing significantly better quality than competing methods, including those with approximate camera pose information. Moreover, our results improve with more views and outperform previous InstantNGP and Gaussian Splatting algorithms even when using half the dataset. Project page: https://raymondjiangkw.github.io/cogs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17142v3">DreamGaussian4D: Generative 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-10
      | 💬 Technical report. Project page is at https://jiawei-ren.github.io/projects/dreamgaussian4d Code is at https://github.com/jiawei-ren/dreamgaussian4d
    </div>
    <details class="paper-abstract">
      4D content generation has achieved remarkable progress recently. However, existing methods suffer from long optimization times, a lack of motion controllability, and a low quality of details. In this paper, we introduce DreamGaussian4D (DG4D), an efficient 4D generation framework that builds on Gaussian Splatting (GS). Our key insight is that combining explicit modeling of spatial transformations with static GS makes an efficient and powerful representation for 4D generation. Moreover, video generation methods have the potential to offer valuable spatial-temporal priors, enhancing the high-quality 4D generation. Specifically, we propose an integral framework with two major modules: 1) Image-to-4D GS - we initially generate static GS with DreamGaussianHD, followed by HexPlane-based dynamic generation with Gaussian deformation; and 2) Video-to-Video Texture Refinement - we refine the generated UV-space texture maps and meanwhile enhance their temporal consistency by utilizing a pre-trained image-to-video diffusion model. Notably, DG4D reduces the optimization time from several hours to just a few minutes, allows the generated 3D motion to be visually controlled, and produces animated meshes that can be realistically rendered in 3D engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06216v1">Lighting Every Darkness with 3DGS: Fast Training and Real-Time Rendering for HDR View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-06-10
    </div>
    <details class="paper-abstract">
      Volumetric rendering based methods, like NeRF, excel in HDR view synthesis from RAWimages, especially for nighttime scenes. While, they suffer from long training times and cannot perform real-time rendering due to dense sampling requirements. The advent of 3D Gaussian Splatting (3DGS) enables real-time rendering and faster training. However, implementing RAW image-based view synthesis directly using 3DGS is challenging due to its inherent drawbacks: 1) in nighttime scenes, extremely low SNR leads to poor structure-from-motion (SfM) estimation in distant views; 2) the limited representation capacity of spherical harmonics (SH) function is unsuitable for RAW linear color space; and 3) inaccurate scene structure hampers downstream tasks such as refocusing. To address these issues, we propose LE3D (Lighting Every darkness with 3DGS). Our method proposes Cone Scatter Initialization to enrich the estimation of SfM, and replaces SH with a Color MLP to represent the RAW linear color space. Additionally, we introduce depth distortion and near-far regularizations to improve the accuracy of scene structure for downstream tasks. These designs enable LE3D to perform real-time novel view synthesis, HDR rendering, refocusing, and tone-mapping changes. Compared to previous volumetric rendering based methods, LE3D reduces training time to 1% and improves rendering speed by up to 4,000 times for 2K resolution images in terms of FPS. Code and viewer can be found in https://github.com/Srameo/LE3D .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09497v1">Simplicits: Mesh-Free, Geometry-Agnostic, Elastic Simulation</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      The proliferation of 3D representations, from explicit meshes to implicit neural fields and more, motivates the need for simulators agnostic to representation. We present a data-, mesh-, and grid-free solution for elastic simulation for any object in any geometric representation undergoing large, nonlinear deformations. We note that every standard geometric representation can be reduced to an occupancy function queried at any point in space, and we define a simulator atop this common interface. For each object, we fit a small implicit neural network encoding spatially varying weights that act as a reduced deformation basis. These weights are trained to learn physically significant motions in the object via random perturbations. Our loss ensures we find a weight-space basis that best minimizes deformation energy by stochastically evaluating elastic energies through Monte Carlo sampling of the deformation volume. At runtime, we simulate in the reduced basis and sample the deformations back to the original domain. Our experiments demonstrate the versatility, accuracy, and speed of this approach on data including signed distance functions, point clouds, neural primitives, tomography scans, radiance fields, Gaussian splats, surface meshes, and volume meshes, as well as showing a variety of material energies, contact models, and time integration schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05852v1">RefGaussian: Disentangling Reflections from 3D Gaussian Splatting for Realistic Rendering</a></div>
    <div class="paper-meta">
      📅 2024-06-09
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has made a notable advancement in the field of neural rendering, 3D scene reconstruction, and novel view synthesis. Nevertheless, 3D-GS encounters the main challenge when it comes to accurately representing physical reflections, especially in the case of total reflection and semi-reflection that are commonly found in real-world scenes. This limitation causes reflections to be mistakenly treated as independent elements with physical presence, leading to imprecise reconstructions. Herein, to tackle this challenge, we propose RefGaussian to disentangle reflections from 3D-GS for realistically modeling reflections. Specifically, we propose to split a scene into transmitted and reflected components and represent these components using two Spherical Harmonics (SH). Given that this decomposition is not fully determined, we employ local regularization techniques to ensure local smoothness for both the transmitted and reflected components, thereby achieving more plausible decomposition outcomes than 3D-GS. Experimental results demonstrate that our approach achieves superior novel view synthesis and accurate depth estimation outcomes. Furthermore, it enables the utilization of scene editing applications, ensuring both high-quality results and physical coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17888v2">2D Gaussian Splatting for Geometrically Accurate Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2024-06-09
      | 💬 13 pages, 12 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently revolutionized radiance field reconstruction, achieving high quality novel view synthesis and fast rendering speed without baking. However, 3DGS fails to accurately represent surfaces due to the multi-view inconsistent nature of 3D Gaussians. We present 2D Gaussian Splatting (2DGS), a novel approach to model and reconstruct geometrically accurate radiance fields from multi-view images. Our key idea is to collapse the 3D volume into a set of 2D oriented planar Gaussian disks. Unlike 3D Gaussians, 2D Gaussians provide view-consistent geometry while modeling surfaces intrinsically. To accurately recover thin surfaces and achieve stable optimization, we introduce a perspective-correct 2D splatting process utilizing ray-splat intersection and rasterization. Additionally, we incorporate depth distortion and normal consistency terms to further enhance the quality of the reconstructions. We demonstrate that our differentiable renderer allows for noise-free and detailed geometry reconstruction while maintaining competitive appearance quality, fast training speed, and real-time rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04343v1">Flash3D: Feed-Forward Generalisable 3D Scene Reconstruction from a Single Image</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Project page: https://www.robots.ox.ac.uk/~vgg/research/flash3d/
    </div>
    <details class="paper-abstract">
      In this paper, we propose Flash3D, a method for scene reconstruction and novel view synthesis from a single image which is both very generalisable and efficient. For generalisability, we start from a "foundation" model for monocular depth estimation and extend it to a full 3D shape and appearance reconstructor. For efficiency, we base this extension on feed-forward Gaussian Splatting. Specifically, we predict a first layer of 3D Gaussians at the predicted depth, and then add additional layers of Gaussians that are offset in space, allowing the model to complete the reconstruction behind occlusions and truncations. Flash3D is very efficient, trainable on a single GPU in a day, and thus accessible to most researchers. It achieves state-of-the-art results when trained and tested on RealEstate10k. When transferred to unseen datasets like NYU it outperforms competitors by a large margin. More impressively, when transferred to KITTI, Flash3D achieves better PSNR than methods trained specifically on that dataset. In some instances, it even outperforms recent methods that use multiple views as input. Code, models, demo, and more results are available at https://www.robots.ox.ac.uk/~vgg/research/flash3d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04253v1">A Survey on 3D Human Avatar Modeling -- From Reconstruction to Generation</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 30 pages, 21 figures
    </div>
    <details class="paper-abstract">
      3D modeling has long been an important area in computer vision and computer graphics. Recently, thanks to the breakthroughs in neural representations and generative models, we witnessed a rapid development of 3D modeling. 3D human modeling, lying at the core of many real-world applications, such as gaming and animation, has attracted significant attention. Over the past few years, a large body of work on creating 3D human avatars has been introduced, forming a new and abundant knowledge base for 3D human modeling. The scale of the literature makes it difficult for individuals to keep track of all the works. This survey aims to provide a comprehensive overview of these emerging techniques for 3D human avatar modeling, from both reconstruction and generation perspectives. Firstly, we review representative methods for 3D human reconstruction, including methods based on pixel-aligned implicit function, neural radiance field, and 3D Gaussian Splatting, etc. We then summarize representative methods for 3D human generation, especially those using large language models like CLIP, diffusion models, and various 3D representations, which demonstrate state-of-the-art performance. Finally, we discuss our reflection on existing methods and open challenges for 3D human avatar modeling, shedding light on future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03697v1">Superpoint Gaussian Splatting for Real-Time High-Fidelity Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Accepted by ICML 2024
    </div>
    <details class="paper-abstract">
      Rendering novel view images in dynamic scenes is a crucial yet challenging task. Current methods mainly utilize NeRF-based methods to represent the static scene and an additional time-variant MLP to model scene deformations, resulting in relatively low rendering quality as well as slow inference speed. To tackle these challenges, we propose a novel framework named Superpoint Gaussian Splatting (SP-GS). Specifically, our framework first employs explicit 3D Gaussians to reconstruct the scene and then clusters Gaussians with similar properties (e.g., rotation, translation, and location) into superpoints. Empowered by these superpoints, our method manages to extend 3D Gaussian splatting to dynamic scenes with only a slight increase in computational expense. Apart from achieving state-of-the-art visual quality and real-time rendering under high resolutions, the superpoint representation provides a stronger manipulation capability. Extensive experiments demonstrate the practicality and effectiveness of our approach on both synthetic and real-world datasets. Please see our project page at https://dnvtmf.github.io/SP_GS.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02541v3">Enhancing Temporal Consistency in Video Editing by Reconstructing Videos with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-06-06
      | 💬 Project page at https://video-3dgs-project.github.io/
    </div>
    <details class="paper-abstract">
      Recent advancements in zero-shot video diffusion models have shown promise for text-driven video editing, but challenges remain in achieving high temporal consistency. To address this, we introduce Video-3DGS, a 3D Gaussian Splatting (3DGS)-based video refiner designed to enhance temporal consistency in zero-shot video editors. Our approach utilizes a two-stage 3D Gaussian optimizing process tailored for editing dynamic monocular videos. In the first stage, Video-3DGS employs an improved version of COLMAP, referred to as MC-COLMAP, which processes original videos using a Masked and Clipped approach. For each video clip, MC-COLMAP generates the point clouds for dynamic foreground objects and complex backgrounds. These point clouds are utilized to initialize two sets of 3D Gaussians (Frg-3DGS and Bkg-3DGS) aiming to represent foreground and background views. Both foreground and background views are then merged with a 2D learnable parameter map to reconstruct full views. In the second stage, we leverage the reconstruction ability developed in the first stage to impose the temporal constraints on the video diffusion model. To demonstrate the efficacy of Video-3DGS on both stages, we conduct extensive experiments across two related tasks: Video Reconstruction and Video Editing. Video-3DGS trained with 3k iterations significantly improves video reconstruction quality (+3 PSNR, +7 PSNR increase) and training efficiency (x1.9, x4.5 times faster) over NeRF-based and 3DGS-based state-of-art methods on DAVIS dataset, respectively. Moreover, it enhances video editing by ensuring temporal consistency across 58 dynamic monocular videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.18454v2">3D Gaussian Splatting with Deferred Reflection</a></div>
    <div class="paper-meta">
      📅 2024-06-05
    </div>
    <details class="paper-abstract">
      The advent of neural and Gaussian-based radiance field methods have achieved great success in the field of novel view synthesis. However, specular reflection remains non-trivial, as the high frequency radiance field is notoriously difficult to fit stably and accurately. We present a deferred shading method to effectively render specular reflection with Gaussian splatting. The key challenge comes from the environment map reflection model, which requires accurate surface normal while simultaneously bottlenecks normal estimation with discontinuous gradients. We leverage the per-pixel reflection gradients generated by deferred shading to bridge the optimization process of neighboring Gaussians, allowing nearly correct normal estimations to gradually propagate and eventually spread over all reflective objects. Our method significantly outperforms state-of-the-art techniques and concurrent work in synthesizing high-quality specular reflection effects, demonstrating a consistent improvement of peak signal-to-noise ratio (PSNR) for both synthetic and real-world scenes, while running at a frame rate almost identical to vanilla Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02533v1">SatSplatYOLO: 3D Gaussian Splatting-based Virtual Object Detection Ensembles for Satellite Feature Recognition</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      On-orbit servicing (OOS), inspection of spacecraft, and active debris removal (ADR). Such missions require precise rendezvous and proximity operations in the vicinity of non-cooperative, possibly unknown, resident space objects. Safety concerns with manned missions and lag times with ground-based control necessitate complete autonomy. In this article, we present an approach for mapping geometries and high-confidence detection of components of unknown, non-cooperative satellites on orbit. We implement accelerated 3D Gaussian splatting to learn a 3D representation of the satellite, render virtual views of the target, and ensemble the YOLOv5 object detector over the virtual views, resulting in reliable, accurate, and precise satellite component detections. The full pipeline capable of running on-board and stand to enable downstream machine intelligence tasks necessary for autonomous guidance, navigation, and control tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02407v1">WE-GS: An In-the-wild Efficient 3D Gaussian Representation for Unconstrained Photo Collections</a></div>
    <div class="paper-meta">
      📅 2024-06-04
      | 💬 Our project page is available at https://yuzewang1998.github.io/we-gs.github.io/
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) from unconstrained photo collections is challenging in computer graphics. Recently, 3D Gaussian Splatting (3DGS) has shown promise for photorealistic and real-time NVS of static scenes. Building on 3DGS, we propose an efficient point-based differentiable rendering framework for scene reconstruction from photo collections. Our key innovation is a residual-based spherical harmonic coefficients transfer module that adapts 3DGS to varying lighting conditions and photometric post-processing. This lightweight module can be pre-computed and ensures efficient gradient propagation from rendered images to 3D Gaussian attributes. Additionally, we observe that the appearance encoder and the transient mask predictor, the two most critical parts of NVS from unconstrained photo collections, can be mutually beneficial. We introduce a plug-and-play lightweight spatial attention module to simultaneously predict transient occluders and latent appearance representation for each image. After training and preprocessing, our method aligns with the standard 3DGS format and rendering pipeline, facilitating seamlessly integration into various 3DGS applications. Extensive experiments on diverse datasets show our approach outperforms existing approaches on the rendering quality of novel view and appearance synthesis with high converge and rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18454v2">3D Gaussian Splatting with Deferred Reflection</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      The advent of neural and Gaussian-based radiance field methods have achieved great success in the field of novel view synthesis. However, specular reflection remains non-trivial, as the high frequency radiance field is notoriously difficult to fit stably and accurately. We present a deferred shading method to effectively render specular reflection with Gaussian splatting. The key challenge comes from the environment map reflection model, which requires accurate surface normal while simultaneously bottlenecks normal estimation with discontinuous gradients. We leverage the per-pixel reflection gradients generated by deferred shading to bridge the optimization process of neighboring Gaussians, allowing nearly correct normal estimations to gradually propagate and eventually spread over all reflective objects. Our method significantly outperforms state-of-the-art techniques and concurrent work in synthesizing high-quality specular reflection effects, demonstrating a consistent improvement of peak signal-to-noise ratio (PSNR) for both synthetic and real-world scenes, while running at a frame rate almost identical to vanilla Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.18454v2">3D Gaussian Splatting with Deferred Reflection</a></div>
    <div class="paper-meta">
      📅 2024-06-04
    </div>
    <details class="paper-abstract">
      The advent of neural and Gaussian-based radiance field methods have achieved great success in the field of novel view synthesis. However, specular reflection remains non-trivial, as the high frequency radiance field is notoriously difficult to fit stably and accurately. We present a deferred shading method to effectively render specular reflection with Gaussian splatting. The key challenge comes from the environment map reflection model, which requires accurate surface normal while simultaneously bottlenecks normal estimation with discontinuous gradients. We leverage the per-pixel reflection gradients generated by deferred shading to bridge the optimization process of neighboring Gaussians, allowing nearly correct normal estimations to gradually propagate and eventually spread over all reflective objects. Our method significantly outperforms state-of-the-art techniques and concurrent work in synthesizing high-quality specular reflection effects, demonstrating a consistent improvement of peak signal-to-noise ratio (PSNR) for both synthetic and real-world scenes, while running at a frame rate almost identical to vanilla Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20310v3">A Pixel Is Worth More Than One 3D Gaussians in Single-View 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-06-03
      | 💬 preprint, under review
    </div>
    <details class="paper-abstract">
      Learning 3D scene representation from a single-view image is a long-standing fundamental problem in computer vision, with the inherent ambiguity in predicting contents unseen from the input view. Built on the recently proposed 3D Gaussian Splatting (3DGS), the Splatter Image method has made promising progress on fast single-image novel view synthesis via learning a single 3D Gaussian for each pixel based on the U-Net feature map of an input image. However, it has limited expressive power to represent occluded components that are not observable in the input view. To address this problem, this paper presents a Hierarchical Splatter Image method in which a pixel is worth more than one 3D Gaussians. Specifically, each pixel is represented by a parent 3D Gaussian and a small number of child 3D Gaussians. Parent 3D Gaussians are learned as done in the vanilla Splatter Image. Child 3D Gaussians are learned via a lightweight Multi-Layer Perceptron (MLP) which takes as input the projected image features of a parent 3D Gaussian and the embedding of a target camera view. Both parent and child 3D Gaussians are learned end-to-end in a stage-wise way. The joint condition of input image features from eyes of the parent Gaussians and the target camera position facilitates learning to allocate child Gaussians to ``see the unseen'', recovering the occluded details that are often missed by parent Gaussians. In experiments, the proposed method is tested on the ShapeNet-SRN and CO3D datasets with state-of-the-art performance obtained, especially showing promising capabilities of reconstructing occluded contents in the input view.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11021v2">Enhanced 3D Urban Scene Reconstruction and Point Cloud Densification using Gaussian Splatting and Google Earth Imagery</a></div>
    <div class="paper-meta">
      📅 2024-06-01
    </div>
    <details class="paper-abstract">
      3D urban scene reconstruction and modelling is a crucial research area in remote sensing with numerous applications in academia, commerce, industry, and administration. Recent advancements in view synthesis models have facilitated photorealistic 3D reconstruction solely from 2D images. Leveraging Google Earth imagery, we construct a 3D Gaussian Splatting model of the Waterloo region centered on the University of Waterloo and are able to achieve view-synthesis results far exceeding previous 3D view-synthesis results based on neural radiance fields which we demonstrate in our benchmark. Additionally, we retrieved the 3D geometry of the scene using the 3D point cloud extracted from the 3D Gaussian Splatting model which we benchmarked against our Multi- View-Stereo dense reconstruction of the scene, thereby reconstructing both the 3D geometry and photorealistic lighting of the large-scale urban scene through 3D Gaussian Splatting
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14455v2">TIGER: Text-Instructed 3D Gaussian Retrieval and Coherent Editing</a></div>
    <div class="paper-meta">
      📅 2024-06-01
    </div>
    <details class="paper-abstract">
      Editing objects within a scene is a critical functionality required across a broad spectrum of applications in computer vision and graphics. As 3D Gaussian Splatting (3DGS) emerges as a frontier in scene representation, the effective modification of 3D Gaussian scenes has become increasingly vital. This process entails accurately retrieve the target objects and subsequently performing modifications based on instructions. Though available in pieces, existing techniques mainly embed sparse semantics into Gaussians for retrieval, and rely on an iterative dataset update paradigm for editing, leading to over-smoothing or inconsistency issues. To this end, this paper proposes a systematic approach, namely TIGER, for coherent text-instructed 3D Gaussian retrieval and editing. In contrast to the top-down language grounding approach for 3D Gaussians, we adopt a bottom-up language aggregation strategy to generate a denser language embedded 3D Gaussians that supports open-vocabulary retrieval. To overcome the over-smoothing and inconsistency issues in editing, we propose a Coherent Score Distillation (CSD) that aggregates a 2D image editing diffusion model and a multi-view diffusion model for score distillation, producing multi-view consistent editing with much finer details. In various experiments, we demonstrate that our TIGER is able to accomplish more consistent and realistic edits than prior work.
    </details>
</div>
