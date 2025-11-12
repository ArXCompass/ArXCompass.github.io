# gaussian splatting - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

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
