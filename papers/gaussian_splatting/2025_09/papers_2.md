# gaussian splatting - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02232v1">Efficient Geometry Compression and Communication for 3D Gaussian Splatting Point Clouds</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 8 pages,5 figures
    </div>
    <details class="paper-abstract">
      Storage and transmission challenges in dynamic 3D scene representation based on the i3DV platform, With increasing scene complexity, the explosive growth of 3D Gaussian data volume causes excessive storage space occupancy. To address this issue, we propose adopting the AVS PCRM reference software for efficient compression of Gaussian point cloud geometry data. The strategy deeply integrates the advanced encoding capabilities of AVS PCRM into the i3DV platform, forming technical complementarity with the original rate-distortion optimization mechanism based on binary hash tables. On one hand, the hash table efficiently caches inter-frame Gaussian point transformation relationships, which allows for high-fidelity transmission within a 40 Mbps bandwidth constraint. On the other hand, AVS PCRM performs precise compression on geometry data. Experimental results demonstrate that the joint framework maintains the advantages of fast rendering and high-quality synthesis in 3D Gaussian technology while achieving significant 10\%-25\% bitrate savings on universal test sets. It provides a superior rate-distortion tradeoff solution for the storage, transmission, and interaction of 3D volumetric video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05220v3">StylizedGS: Controllable Stylization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 [TPAMI 2025] Project Page: https://kristen-z.github.io/stylizedgs/
    </div>
    <details class="paper-abstract">
      As XR technology continues to advance rapidly, 3D generation and editing are increasingly crucial. Among these, stylization plays a key role in enhancing the appearance of 3D models. By utilizing stylization, users can achieve consistent artistic effects in 3D editing using a single reference style image, making it a user-friendly editing method. However, recent NeRF-based 3D stylization methods encounter efficiency issues that impact the user experience, and their implicit nature limits their ability to accurately transfer geometric pattern styles. Additionally, the ability for artists to apply flexible control over stylized scenes is considered highly desirable to foster an environment conducive to creative exploration. To address the above issues, we introduce StylizedGS, an efficient 3D neural style transfer framework with adaptable control over perceptual factors based on 3D Gaussian Splatting representation. We propose a filter-based refinement to eliminate floaters that affect the stylization effects in the scene reconstruction process. The nearest neighbor-based style loss is introduced to achieve stylization by fine-tuning the geometry and color parameters of 3DGS, while a depth preservation loss with other regularizations is proposed to prevent the tampering of geometry content. Moreover, facilitated by specially designed losses, StylizedGS enables users to control color, stylized scale, and regions during the stylization to possess customization capabilities. Our method achieves high-quality stylization results characterized by faithful brushstrokes and geometric consistency with flexible controls. Extensive experiments across various scenes and styles demonstrate the effectiveness and efficiency of our method concerning both stylization quality and inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02141v1">GRMM: Real-Time High-Fidelity Gaussian Morphable Head Model with Learned Residuals</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 Project page: https://mohitm1994.github.io/GRMM/
    </div>
    <details class="paper-abstract">
      3D Morphable Models (3DMMs) enable controllable facial geometry and expression editing for reconstruction, animation, and AR/VR, but traditional PCA-based mesh models are limited in resolution, detail, and photorealism. Neural volumetric methods improve realism but remain too slow for interactive use. Recent Gaussian Splatting (3DGS) based facial models achieve fast, high-quality rendering but still depend solely on a mesh-based 3DMM prior for expression control, limiting their ability to capture fine-grained geometry, expressions, and full-head coverage. We introduce GRMM, the first full-head Gaussian 3D morphable model that augments a base 3DMM with residual geometry and appearance components, additive refinements that recover high-frequency details such as wrinkles, fine skin texture, and hairline variations. GRMM provides disentangled control through low-dimensional, interpretable parameters (e.g., identity shape, facial expressions) while separately modelling residuals that capture subject- and expression-specific detail beyond the base model's capacity. Coarse decoders produce vertex-level mesh deformations, fine decoders represent per-Gaussian appearance, and a lightweight CNN refines rasterised images for enhanced realism, all while maintaining 75 FPS real-time rendering. To learn consistent, high-fidelity residuals, we present EXPRESS-50, the first dataset with 60 aligned expressions across 50 identities, enabling robust disentanglement of identity and expression in Gaussian-based 3DMMs. Across monocular 3D face reconstruction, novel-view synthesis, and expression transfer, GRMM surpasses state-of-the-art methods in fidelity and expression accuracy while delivering interactive real-time performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01964v1">2D Gaussian Splatting with Semantic Alignment for Image Inpainting</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS), a recent technique for converting discrete points into continuous spatial representations, has shown promising results in 3D scene modeling and 2D image super-resolution. In this paper, we explore its untapped potential for image inpainting, which demands both locally coherent pixel synthesis and globally consistent semantic restoration. We propose the first image inpainting framework based on 2D Gaussian Splatting, which encodes incomplete images into a continuous field of 2D Gaussian splat coefficients and reconstructs the final image via a differentiable rasterization process. The continuous rendering paradigm of GS inherently promotes pixel-level coherence in the inpainted results. To improve efficiency and scalability, we introduce a patch-wise rasterization strategy that reduces memory overhead and accelerates inference. For global semantic consistency, we incorporate features from a pretrained DINO model. We observe that DINO's global features are naturally robust to small missing regions and can be effectively adapted to guide semantic alignment in large-mask scenarios, ensuring that the inpainted content remains contextually consistent with the surrounding scene. Extensive experiments on standard benchmarks demonstrate that our method achieves competitive performance in both quantitative metrics and perceptual quality, establishing a new direction for applying Gaussian Splatting to 2D image processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09573v2">FreeSplatter: Pose-free Gaussian Splatting for Sparse-view 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Project page: https://bluestyle97.github.io/projects/freesplatter/
    </div>
    <details class="paper-abstract">
      Sparse-view reconstruction models typically require precise camera poses, yet obtaining these parameters from sparse-view images remains challenging. We introduce FreeSplatter, a scalable feed-forward framework that generates high-quality 3D Gaussians from uncalibrated sparse-view images while estimating camera parameters within seconds. Our approach employs a streamlined transformer architecture where self-attention blocks facilitate information exchange among multi-view image tokens, decoding them into pixel-aligned 3D Gaussian primitives within a unified reference frame. This representation enables both high-fidelity 3D modeling and efficient camera parameter estimation using off-the-shelf solvers. We develop two specialized variants--for object-centric and scene-level reconstruction--trained on comprehensive datasets. Remarkably, FreeSplatter outperforms several pose-dependent Large Reconstruction Models (LRMs) by a notable margin while achieving comparable or even better pose estimation accuracy compared to state-of-the-art pose-free reconstruction approach MASt3R in challenging benchmarks. Beyond technical benchmarks, FreeSplatter streamlines text/image-to-3D content creation pipelines, eliminating the complexity of camera pose management while delivering exceptional visual fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10462v2">BloomScene: Lightweight Structured 3D Gaussian Splatting for Crossmodal Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by AAAI 2025. Code: https://github.com/SparklingH/BloomScene
    </div>
    <details class="paper-abstract">
      With the widespread use of virtual reality applications, 3D scene generation has become a new challenging research frontier. 3D scenes have highly complex structures and need to ensure that the output is dense, coherent, and contains all necessary structures. Many current 3D scene generation methods rely on pre-trained text-to-image diffusion models and monocular depth estimators. However, the generated scenes occupy large amounts of storage space and often lack effective regularisation methods, leading to geometric distortions. To this end, we propose BloomScene, a lightweight structured 3D Gaussian splatting for crossmodal scene generation, which creates diverse and high-quality 3D scenes from text or image inputs. Specifically, a crossmodal progressive scene generation framework is proposed to generate coherent scenes utilizing incremental point cloud reconstruction and 3D Gaussian splatting. Additionally, we propose a hierarchical depth prior-based regularization mechanism that utilizes multi-level constraints on depth accuracy and smoothness to enhance the realism and continuity of the generated scenes. Ultimately, we propose a structured context-guided compression mechanism that exploits structured hash grids to model the context of unorganized anchor attributes, which significantly eliminates structural redundancy and reduces storage overhead. Comprehensive experiments across multiple scenes demonstrate the significant potential and advantages of our framework compared with several baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01681v1">GaussianGAN: Real-Time Photorealistic controllable Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 IEEE conference series on Automatic Face and Gesture Recognition 2025
    </div>
    <details class="paper-abstract">
      Photorealistic and controllable human avatars have gained popularity in the research community thanks to rapid advances in neural rendering, providing fast and realistic synthesis tools. However, a limitation of current solutions is the presence of noticeable blurring. To solve this problem, we propose GaussianGAN, an animatable avatar approach developed for photorealistic rendering of people in real-time. We introduce a novel Gaussian splatting densification strategy to build Gaussian points from the surface of cylindrical structures around estimated skeletal limbs. Given the camera calibration, we render an accurate semantic segmentation with our novel view segmentation module. Finally, a UNet generator uses the rendered Gaussian splatting features and the segmentation maps to create photorealistic digital avatars. Our method runs in real-time with a rendering speed of 79 FPS. It outperforms previous methods regarding visual perception and quality, achieving a state-of-the-art results in terms of a pixel fidelity of 32.94db on the ZJU Mocap dataset and 33.39db on the Thuman4 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01469v1">Im2Haircut: Single-view Strand-based Hair Reconstruction for Human Avatars</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 For more results please refer to the project page https://im2haircut.is.tue.mpg.de
    </div>
    <details class="paper-abstract">
      We present a novel approach for 3D hair reconstruction from single photographs based on a global hair prior combined with local optimization. Capturing strand-based hair geometry from single photographs is challenging due to the variety and geometric complexity of hairstyles and the lack of ground truth training data. Classical reconstruction methods like multi-view stereo only reconstruct the visible hair strands, missing the inner structure of hairstyles and hampering realistic hair simulation. To address this, existing methods leverage hairstyle priors trained on synthetic data. Such data, however, is limited in both quantity and quality since it requires manual work from skilled artists to model the 3D hairstyles and create near-photorealistic renderings. To address this, we propose a novel approach that uses both, real and synthetic data to learn an effective hairstyle prior. Specifically, we train a transformer-based prior model on synthetic data to obtain knowledge of the internal hairstyle geometry and introduce real data in the learning process to model the outer structure. This training scheme is able to model the visible hair strands depicted in an input image, while preserving the general 3D structure of hairstyles. We exploit this prior to create a Gaussian-splatting-based reconstruction method that creates hairstyles from one or more images. Qualitative and quantitative comparisons with existing reconstruction pipelines demonstrate the effectiveness and superior performance of our method for capturing detailed hair orientation, overall silhouette, and backside consistency. For additional results and code, please refer to https://im2haircut.is.tue.mpg.de.
    </details>
</div>
