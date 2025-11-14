# gaussian splatting - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01823v1">HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project Page: https://timsong412.github.io/HDGS-ProjPage/
    </div>
    <details class="paper-abstract">
      Recent advancements in neural rendering, particularly 2D Gaussian Splatting (2DGS), have shown promising results for jointly reconstructing fine appearance and geometry by leveraging 2D Gaussian surfels. However, current methods face significant challenges when rendering at arbitrary viewpoints, such as anti-aliasing for down-sampled rendering, and texture detail preservation for high-resolution rendering. We proposed a novel method to align the 2D surfels with texture maps and augment it with per-ray depth sorting and fisher-based pruning for rendering consistency and efficiency. With correct order, per-surfel texture maps significantly improve the capabilities to capture fine details. Additionally, to render high-fidelity details in varying viewpoints, we designed a frustum-based sampling method to mitigate the aliasing artifacts. Experimental results on benchmarks and our custom texture-rich dataset demonstrate that our method surpasses existing techniques, particularly in detail preservation and anti-aliasing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01807v1">Occam's LGS: A Simple Approach for Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project Page: https://insait-institute.github.io/OccamLGS/
    </div>
    <details class="paper-abstract">
      TL;DR: Gaussian Splatting is a widely adopted approach for 3D scene representation that offers efficient, high-quality 3D reconstruction and rendering. A major reason for the success of 3DGS is its simplicity of representing a scene with a set of Gaussians, which makes it easy to interpret and adapt. To enhance scene understanding beyond the visual representation, approaches have been developed that extend 3D Gaussian Splatting with semantic vision-language features, especially allowing for open-set tasks. In this setting, the language features of 3D Gaussian Splatting are often aggregated from multiple 2D views. Existing works address this aggregation problem using cumbersome techniques that lead to high computational cost and training time. In this work, we show that the sophisticated techniques for language-grounded 3D Gaussian Splatting are simply unnecessary. Instead, we apply Occam's razor to the task at hand and perform weighted multi-view feature aggregation using the weights derived from the standard rendering process, followed by a simple heuristic-based noisy Gaussian filtration. Doing so offers us state-of-the-art results with a speed-up of two orders of magnitude. We showcase our results in two commonly used benchmark datasets: LERF and 3D-OVS. Our simple approach allows us to perform reasoning directly in the language features, without any compression whatsoever. Such modeling in turn offers easy scene manipulation, unlike the existing methods -- which we illustrate using an application of object insertion in the scene. Furthermore, we provide a thorough discussion regarding the significance of our contributions within the context of the current literature. Project Page: https://insait-institute.github.io/OccamLGS/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01792v1">CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://ihe-kaii.github.io/CTRL-D/
    </div>
    <details class="paper-abstract">
      Recent advances in 3D representations, such as Neural Radiance Fields and 3D Gaussian Splatting, have greatly improved realistic scene modeling and novel-view synthesis. However, achieving controllable and consistent editing in dynamic 3D scenes remains a significant challenge. Previous work is largely constrained by its editing backbones, resulting in inconsistent edits and limited controllability. In our work, we introduce a novel framework that first fine-tunes the InstructPix2Pix model, followed by a two-stage optimization of the scene based on deformable 3D Gaussians. Our fine-tuning enables the model to "learn" the editing ability from a single edited reference image, transforming the complex task of dynamic scene editing into a simple 2D image editing process. By directly learning editing regions and styles from the reference, our approach enables consistent and precise local edits without the need for tracking desired editing regions, effectively addressing key challenges in dynamic scene editing. Then, our two-stage optimization progressively edits the trained dynamic scene, using a designed edited image buffer to accelerate convergence and improve temporal consistency. Compared to state-of-the-art methods, our approach offers more flexible and controllable local scene editing, achieving high-quality and consistent results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19895v2">GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://narcissusex.github.io/GuardSplat and Code: https://github.com/NarcissusEx/GuardSplat
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently created impressive assets for various applications. However, the copyright of these assets is not well protected as existing watermarking methods are not suited for 3DGS considering security, capacity, and invisibility. Besides, these methods often require hours or even days for optimization, limiting the application scenarios. In this paper, we propose GuardSplat, an innovative and efficient framework that effectively protects the copyright of 3DGS assets. Specifically, 1) We first propose a CLIP-guided Message Decoupling Optimization module for training the message decoder, leveraging CLIP's aligning capability and rich representations to achieve a high extraction accuracy with minimal optimization costs, presenting exceptional capability and efficiency. 2) Then, we propose a Spherical-harmonic-aware (SH-aware) Message Embedding module tailored for 3DGS, which employs a set of SH offsets to seamlessly embed the message into the SH features of each 3D Gaussian while maintaining the original 3D structure. It enables the 3DGS assets to be watermarked with minimal fidelity trade-offs and prevents malicious users from removing the messages from the model files, meeting the demands for invisibility and security. 3) We further propose an Anti-distortion Message Extraction module to improve robustness against various visual distortions. Extensive experiments demonstrate that GuardSplat outperforms the state-of-the-art methods and achieves fast optimization speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01745v1">Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Seamless integration of both aerial and street view images remains a significant challenge in neural scene reconstruction and rendering. Existing methods predominantly focus on single domain, limiting their applications in immersive environments, which demand extensive free view exploration with large view changes both horizontally and vertically. We introduce Horizon-GS, a novel approach built upon Gaussian Splatting techniques, tackles the unified reconstruction and rendering for aerial and street views. Our method addresses the key challenges of combining these perspectives with a new training strategy, overcoming viewpoint discrepancies to generate high-fidelity scenes. We also curate a high-quality aerial-to-ground views dataset encompassing both synthetic and real-world scene to advance further research. Experiments across diverse urban scene datasets confirm the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01718v1">HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Our project page is at https://xdimlab.github.io/HUGSIM
    </div>
    <details class="paper-abstract">
      In the past few decades, autonomous driving algorithms have made significant progress in perception, planning, and control. However, evaluating individual components does not fully reflect the performance of entire systems, highlighting the need for more holistic assessment methods. This motivates the development of HUGSIM, a closed-loop, photo-realistic, and real-time simulator for evaluating autonomous driving algorithms. We achieve this by lifting captured 2D RGB images into the 3D space via 3D Gaussian Splatting, improving the rendering quality for closed-loop scenarios, and building the closed-loop environment. In terms of rendering, We tackle challenges of novel view synthesis in closed-loop scenarios, including viewpoint extrapolation and 360-degree vehicle rendering. Beyond novel view synthesis, HUGSIM further enables the full closed simulation loop, dynamically updating the ego and actor states and observations based on control commands. Moreover, HUGSIM offers a comprehensive benchmark across more than 70 sequences from KITTI-360, Waymo, nuScenes, and PandaSet, along with over 400 varying scenarios, providing a fair and realistic evaluation platform for existing autonomous driving algorithms. HUGSIM not only serves as an intuitive evaluation benchmark but also unlocks the potential for fine-tuning autonomous driving algorithms in a photorealistic closed-loop setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01717v1">Driving Scene Synthesis on Free-form Trajectories with Generative Prior</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Driving scene synthesis along free-form trajectories is essential for driving simulations to enable closed-loop evaluation of end-to-end driving policies. While existing methods excel at novel view synthesis on recorded trajectories, they face challenges with novel trajectories due to limited views of driving videos and the vastness of driving environments. To tackle this challenge, we propose a novel free-form driving view synthesis approach, dubbed DriveX, by leveraging video generative prior to optimize a 3D model across a variety of trajectories. Concretely, we crafted an inverse problem that enables a video diffusion model to be utilized as a prior for many-trajectory optimization of a parametric 3D model (e.g., Gaussian splatting). To seamlessly use the generative prior, we iteratively conduct this process during optimization. Our resulting model can produce high-fidelity virtual driving environments outside the recorded trajectory, enabling free-form trajectory driving simulation. Beyond real driving scenes, DriveX can also be utilized to simulate virtual driving worlds from AI-generated videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06390v2">Deepfake for the Good: Generating Avatars through Face-Swapping with Implicit Deepfake Generation</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Numerous emerging deep-learning techniques have had a substantial impact on computer graphics. Among the most promising breakthroughs are the rise of Neural Radiance Fields (NeRFs) and Gaussian Splatting (GS). NeRFs encode the object's shape and color in neural network weights using a handful of images with known camera positions to generate novel views. In contrast, GS provides accelerated training and inference without a decrease in rendering quality by encoding the object's characteristics in a collection of Gaussian distributions. These two techniques have found many use cases in spatial computing and other domains. On the other hand, the emergence of deepfake methods has sparked considerable controversy. Deepfakes refers to artificial intelligence-generated videos that closely mimic authentic footage. Using generative models, they can modify facial features, enabling the creation of altered identities or expressions that exhibit a remarkably realistic appearance to a real person. Despite these controversies, deepfake can offer a next-generation solution for avatar creation and gaming when of desirable quality. To that end, we show how to combine all these emerging technologies to obtain a more plausible outcome. Our ImplicitDeepfake uses the classical deepfake algorithm to modify all training images separately and then train NeRF and GS on modified faces. Such simple strategies can produce plausible 3D deepfake-based avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01553v1">SfM-Free 3D Gaussian Splatting via Hierarchical Training</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Standard 3D Gaussian Splatting (3DGS) relies on known or pre-computed camera poses and a sparse point cloud, obtained from structure-from-motion (SfM) preprocessing, to initialize and grow 3D Gaussians. We propose a novel SfM-Free 3DGS (SFGS) method for video input, eliminating the need for known camera poses and SfM preprocessing. Our approach introduces a hierarchical training strategy that trains and merges multiple 3D Gaussian representations -- each optimized for specific scene regions -- into a single, unified 3DGS model representing the entire scene. To compensate for large camera motions, we leverage video frame interpolation models. Additionally, we incorporate multi-source supervision to reduce overfitting and enhance representation. Experimental results reveal that our approach significantly surpasses state-of-the-art SfM-free novel view synthesis methods. On the Tanks and Temples dataset, we improve PSNR by an average of 2.25dB, with a maximum gain of 3.72dB in the best scene. On the CO3D-V2 dataset, we achieve an average PSNR boost of 1.74dB, with a top gain of 3.90dB. The code is available at https://github.com/jibo27/3DGS_Hierarchical_Training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01543v1">6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Efficient and accurate object pose estimation is an essential component for modern vision systems in many applications such as Augmented Reality, autonomous driving, and robotics. While research in model-based 6D object pose estimation has delivered promising results, model-free methods are hindered by the high computational load in rendering and inferring consistent poses of arbitrary objects in a live RGB-D video stream. To address this issue, we present 6DOPE-GS, a novel method for online 6D object pose estimation \& tracking with a single RGB-D camera by effectively leveraging advances in Gaussian Splatting. Thanks to the fast differentiable rendering capabilities of Gaussian Splatting, 6DOPE-GS can simultaneously optimize for 6D object poses and 3D object reconstruction. To achieve the necessary efficiency and accuracy for live tracking, our method uses incremental 2D Gaussian Splatting with an intelligent dynamic keyframe selection procedure to achieve high spatial object coverage and prevent erroneous pose updates. We also propose an opacity statistic-based pruning mechanism for adaptive Gaussian density control, to ensure training stability and efficiency. We evaluate our method on the HO3D and YCBInEOAT datasets and show that 6DOPE-GS matches the performance of state-of-the-art baselines for model-free simultaneous 6D pose tracking and reconstruction while providing a 5$\times$ speedup. We also demonstrate the method's suitability for live, dynamic object tracking and reconstruction in a real-world setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12440v3">Beyond Gaussians: Fast and High-Fidelity 3D Splatting with Linear Kernels</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS) have substantially improved novel view synthesis, enabling high-quality reconstruction and real-time rendering. However, blurring artifacts, such as floating primitives and over-reconstruction, remain challenging. Current methods address these issues by refining scene structure, enhancing geometric representations, addressing blur in training images, improving rendering consistency, and optimizing density control, yet the role of kernel design remains underexplored. We identify the soft boundaries of Gaussian ellipsoids as one of the causes of these artifacts, limiting detail capture in high-frequency regions. To bridge this gap, we introduce 3D Linear Splatting (3DLS), which replaces Gaussian kernels with linear kernels to achieve sharper and more precise results, particularly in high-frequency regions. Through evaluations on three datasets, 3DLS demonstrates state-of-the-art fidelity and accuracy, along with a 30% FPS improvement over baseline 3DGS. The implementation will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01402v1">ULSR-GS: Ultra Large-scale Surface Reconstruction Gaussian Splatting with Multi-View Geometric Consistency</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://ulsrgs.github.io
    </div>
    <details class="paper-abstract">
      While Gaussian Splatting (GS) demonstrates efficient and high-quality scene rendering and small area surface extraction ability, it falls short in handling large-scale aerial image surface extraction tasks. To overcome this, we present ULSR-GS, a framework dedicated to high-fidelity surface extraction in ultra-large-scale scenes, addressing the limitations of existing GS-based mesh extraction methods. Specifically, we propose a point-to-photo partitioning approach combined with a multi-view optimal view matching principle to select the best training images for each sub-region. Additionally, during training, ULSR-GS employs a densification strategy based on multi-view geometric consistency to enhance surface extraction details. Experimental results demonstrate that ULSR-GS outperforms other state-of-the-art GS-based works on large-scale aerial photogrammetry benchmark datasets, significantly improving surface extraction accuracy in complex urban environments. Project page: https://ulsrgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19454v2">GausSurf: Geometry-Guided 3D Gaussian Splatting for Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 Project page: https://jiepengwang.github.io/GausSurf/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has achieved impressive performance in novel view synthesis with real-time rendering capabilities. However, reconstructing high-quality surfaces with fine details using 3D Gaussians remains a challenging task. In this work, we introduce GausSurf, a novel approach to high-quality surface reconstruction by employing geometry guidance from multi-view consistency in texture-rich areas and normal priors in texture-less areas of a scene. We observe that a scene can be mainly divided into two primary regions: 1) texture-rich and 2) texture-less areas. To enforce multi-view consistency at texture-rich areas, we enhance the reconstruction quality by incorporating a traditional patch-match based Multi-View Stereo (MVS) approach to guide the geometry optimization in an iterative scheme. This scheme allows for mutual reinforcement between the optimization of Gaussians and patch-match refinement, which significantly improves the reconstruction results and accelerates the training process. Meanwhile, for the texture-less areas, we leverage normal priors from a pre-trained normal estimation model to guide optimization. Extensive experiments on the DTU and Tanks and Temples datasets demonstrate that our method surpasses state-of-the-art methods in terms of reconstruction quality and computation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16323v2">LeanGaussian: Breaking Pixel or Point Cloud Correspondence in Modeling 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Rencently, Gaussian splatting has demonstrated significant success in novel view synthesis. Current methods often regress Gaussians with pixel or point cloud correspondence, linking each Gaussian with a pixel or a 3D point. This leads to the redundancy of Gaussians being used to overfit the correspondence rather than the objects represented by the 3D Gaussians themselves, consequently wasting resources and lacking accurate geometries or textures. In this paper, we introduce LeanGaussian, a novel approach that treats each query in deformable Transformer as one 3D Gaussian ellipsoid, breaking the pixel or point cloud correspondence constraints. We leverage deformable decoder to iteratively refine the Gaussians layer-by-layer with the image features as keys and values. Notably, the center of each 3D Gaussian is defined as 3D reference points, which are then projected onto the image for deformable attention in 2D space. On both the ShapeNet SRN dataset (category level) and the Google Scanned Objects dataset (open-category level, trained with the Objaverse dataset), our approach, outperforms prior methods by approximately 6.1\%, achieving a PSNR of 25.44 and 22.36, respectively. Additionally, our method achieves a 3D reconstruction speed of 7.2 FPS and rendering speed 500 FPS. The code will be released at https://github.com/jwubz123/DIG3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.10625v2">Gaussian Splatting Decoder for 3D-aware Generative Adversarial Networks</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 CVPRW
    </div>
    <details class="paper-abstract">
      NeRF-based 3D-aware Generative Adversarial Networks (GANs) like EG3D or GIRAFFE have shown very high rendering quality under large representational variety. However, rendering with Neural Radiance Fields poses challenges for 3D applications: First, the significant computational demands of NeRF rendering preclude its use on low-power devices, such as mobiles and VR/AR headsets. Second, implicit representations based on neural networks are difficult to incorporate into explicit 3D scenes, such as VR environments or video games. 3D Gaussian Splatting (3DGS) overcomes these limitations by providing an explicit 3D representation that can be rendered efficiently at high frame rates. In this work, we present a novel approach that combines the high rendering quality of NeRF-based 3D-aware GANs with the flexibility and computational advantages of 3DGS. By training a decoder that maps implicit NeRF representations to explicit 3D Gaussian Splatting attributes, we can integrate the representational diversity and quality of 3D GANs into the ecosystem of 3D Gaussian Splatting for the first time. Additionally, our approach allows for a high resolution GAN inversion and real-time GAN editing with 3D Gaussian Splatting scenes. Project page: florian-barthel.github.io/gaussian_decoder
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.19594v1">Tortho-Gaussian: Splatting True Digital Orthophoto Maps</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 This work has been submitted to the IEEE Transactions on Geoscience and Remote Sensing for possible publication
    </div>
    <details class="paper-abstract">
      True Digital Orthophoto Maps (TDOMs) are essential products for digital twins and Geographic Information Systems (GIS). Traditionally, TDOM generation involves a complex set of traditional photogrammetric process, which may deteriorate due to various challenges, including inaccurate Digital Surface Model (DSM), degenerated occlusion detections, and visual artifacts in weak texture regions and reflective surfaces, etc. To address these challenges, we introduce TOrtho-Gaussian, a novel method inspired by 3D Gaussian Splatting (3DGS) that generates TDOMs through orthogonal splatting of optimized anisotropic Gaussian kernel. More specifically, we first simplify the orthophoto generation by orthographically splatting the Gaussian kernels onto 2D image planes, formulating a geometrically elegant solution that avoids the need for explicit DSM and occlusion detection. Second, to produce TDOM of large-scale area, a divide-and-conquer strategy is adopted to optimize memory usage and time efficiency of training and rendering for 3DGS. Lastly, we design a fully anisotropic Gaussian kernel that adapts to the varying characteristics of different regions, particularly improving the rendering quality of reflective surfaces and slender structures. Extensive experimental evaluations demonstrate that our method outperforms existing commercial software in several aspects, including the accuracy of building boundaries, the visual quality of low-texture regions and building facades. These results underscore the potential of our approach for large-scale urban scene reconstruction, offering a robust alternative for enhancing TDOM quality and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.01459v4">GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) is a novel, state-of-the-art technique for rendering points in a 3D scene by approximating their contribution to image pixels through Gaussian distributions, warranting fast training and real-time rendering. The main drawback of GS is the absence of a well-defined approach for its conditioning due to the necessity of conditioning several hundred thousand Gaussian components. To solve this, we introduce the Gaussian Mesh Splatting (GaMeS) model, which allows modification of Gaussian components in a similar way as meshes. We parameterize each Gaussian component by the vertices of the mesh face. Furthermore, our model needs mesh initialization on input or estimated mesh during training. We also define Gaussian splats solely based on their location on the mesh, allowing for automatic adjustments in position, scale, and rotation during animation. As a result, we obtain a real-time rendering of editable GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.01931v1">Planar Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-02
      | 💬 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025
    </div>
    <details class="paper-abstract">
      This paper presents Planar Gaussian Splatting (PGS), a novel neural rendering approach to learn the 3D geometry and parse the 3D planes of a scene, directly from multiple RGB images. The PGS leverages Gaussian primitives to model the scene and employ a hierarchical Gaussian mixture approach to group them. Similar Gaussians are progressively merged probabilistically in the tree-structured Gaussian mixtures to identify distinct 3D plane instances and form the overall 3D scene geometry. In order to enable the grouping, the Gaussian primitives contain additional parameters, such as plane descriptors derived by lifting 2D masks from a general 2D segmentation model and surface normals. Experiments show that the proposed PGS achieves state-of-the-art performance in 3D planar reconstruction without requiring either 3D plane labels or depth supervision. In contrast to existing supervised methods that have limited generalizability and struggle under domain shift, PGS maintains its performance across datasets thanks to its neural rendering and scene-specific optimization mechanism, while also being significantly faster than existing optimization-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00905v1">Ref-GS: Directional Factorization for 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 Project page: https://ref-gs.github.io/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Ref-GS, a novel approach for directional light factorization in 2D Gaussian splatting, which enables photorealistic view-dependent appearance rendering and precise geometry recovery. Ref-GS builds upon the deferred rendering of Gaussian splatting and applies directional encoding to the deferred-rendered surface, effectively reducing the ambiguity between orientation and viewing angle. Next, we introduce a spherical Mip-grid to capture varying levels of surface roughness, enabling roughness-aware Gaussian shading. Additionally, we propose a simple yet efficient geometry-lighting factorization that connects geometry and lighting via the vector outer product, significantly reducing renderer overhead when integrating volumetric attributes. Our method achieves superior photorealistic rendering for a range of open-world scenes while also accurately recovering geometry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00851v1">DynSUP: Dynamic Gaussian Splatting from An Unposed Image Pair</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting have shown promising results. Existing methods typically assume static scenes and/or multiple images with prior poses. Dynamics, sparse views, and unknown poses significantly increase the problem complexity due to insufficient geometric constraints. To overcome this challenge, we propose a method that can use only two images without prior poses to fit Gaussians in dynamic environments. To achieve this, we introduce two technical contributions. First, we propose an object-level two-view bundle adjustment. This strategy decomposes dynamic scenes into piece-wise rigid components, and jointly estimates the camera pose and motions of dynamic objects. Second, we design an SE(3) field-driven Gaussian training method. It enables fine-grained motion modeling through learnable per-Gaussian transformations. Our method leads to high-fidelity novel view synthesis of dynamic scenes while accurately preserving temporal consistency and object motion. Experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms state-of-the-art approaches designed for the cases of static environments, multiple images, and/or known poses. Our project page is available at https://colin-de.github.io/DynSUP/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00734v1">ChatSplat: 3D Conversational Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-01
    </div>
    <details class="paper-abstract">
      Humans naturally interact with their 3D surroundings using language, and modeling 3D language fields for scene understanding and interaction has gained growing interest. This paper introduces ChatSplat, a system that constructs a 3D language field, enabling rich chat-based interaction within 3D space. Unlike existing methods that primarily use CLIP-derived language features focused solely on segmentation, ChatSplat facilitates interaction on three levels: objects, views, and the entire 3D scene. For view-level interaction, we designed an encoder that encodes the rendered feature map of each view into tokens, which are then processed by a large language model (LLM) for conversation. At the scene level, ChatSplat combines multi-view tokens, enabling interactions that consider the entire scene. For object-level interaction, ChatSplat uses a patch-wise language embedding, unlike LangSplat's pixel-wise language embedding that implicitly includes mask and embedding. Here, we explicitly decouple the language embedding into separate mask and feature map representations, allowing more flexible object-level interaction. To address the challenge of learning 3D Gaussians posed by the complex and diverse distribution of language embeddings used in the LLM, we introduce a learnable normalization technique to standardize these embeddings, facilitating effective learning. Extensive experimental results demonstrate that ChatSplat supports multi-level interactions -- object, view, and scene -- within 3D space, enhancing both understanding and engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00682v1">FlashSLAM: Accelerated RGB-D SLAM for Real-Time 3D Scene Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-12-01
      | 💬 16 pages, 9 figures, 13 tables
    </div>
    <details class="paper-abstract">
      We present FlashSLAM, a novel SLAM approach that leverages 3D Gaussian Splatting for efficient and robust 3D scene reconstruction. Existing 3DGS-based SLAM methods often fall short in sparse view settings and during large camera movements due to their reliance on gradient descent-based optimization, which is both slow and inaccurate. FlashSLAM addresses these limitations by combining 3DGS with a fast vision-based camera tracking technique, utilizing a pretrained feature matching model and point cloud registration for precise pose estimation in under 80 ms - a 90% reduction in tracking time compared to SplaTAM - without costly iterative rendering. In sparse settings, our method achieves up to a 92% improvement in average tracking accuracy over previous methods. Additionally, it accounts for noise in depth sensors, enhancing robustness when using unspecialized devices such as smartphones. Extensive experiments show that FlashSLAM performs reliably across both sparse and dense settings, in synthetic and real-world environments. Evaluations on benchmark datasets highlight its superior accuracy and efficiency, establishing FlashSLAM as a versatile and high-performance solution for SLAM, advancing the state-of-the-art in 3D reconstruction across diverse applications.
    </details>
</div>
