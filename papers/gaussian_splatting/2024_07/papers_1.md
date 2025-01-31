# gaussian splatting - 2024_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00083v1">Localized Gaussian Splatting Editing with Contextual Awareness</a></div>
    <div class="paper-meta">
      📅 2024-07-31
    </div>
    <details class="paper-abstract">
      Recent text-guided generation of individual 3D object has achieved great success using diffusion priors. However, these methods are not suitable for object insertion and replacement tasks as they do not consider the background, leading to illumination mismatches within the environment. To bridge the gap, we introduce an illumination-aware 3D scene editing pipeline for 3D Gaussian Splatting (3DGS) representation. Our key observation is that inpainting by the state-of-the-art conditional 2D diffusion model is consistent with background in lighting. To leverage the prior knowledge from the well-trained diffusion models for 3D object generation, our approach employs a coarse-to-fine objection optimization pipeline with inpainted views. In the first coarse step, we achieve image-to-3D lifting given an ideal inpainted view. The process employs 3D-aware diffusion prior from a view-conditioned diffusion model, which preserves illumination present in the conditioning image. To acquire an ideal inpainted image, we introduce an Anchor View Proposal (AVP) algorithm to find a single view that best represents the scene illumination in target region. In the second Texture Enhancement step, we introduce a novel Depth-guided Inpainting Score Distillation Sampling (DI-SDS), which enhances geometry and texture details with the inpainting diffusion prior, beyond the scope of the 3D-aware diffusion prior knowledge in the first coarse step. DI-SDS not only provides fine-grained texture enhancement, but also urges optimization to respect scene lighting. Our approach efficiently achieves local editing with global illumination consistency without explicitly modeling light transport. We demonstrate robustness of our method by evaluating editing in real scenes containing explicit highlight and shadows, and compare against the state-of-the-art text-to-3D editing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21686v1">Expressive Whole-Body 3D Gaussian Avatar</a></div>
    <div class="paper-meta">
      📅 2024-07-31
      | 💬 Accepted to ECCV 2024. Project page: https://mks0601.github.io/ExAvatar/
    </div>
    <details class="paper-abstract">
      Facial expression and hand motions are necessary to express our emotions and interact with the world. Nevertheless, most of the 3D human avatars modeled from a casually captured video only support body motions without facial expressions and hand motions.In this work, we present ExAvatar, an expressive whole-body 3D human avatar learned from a short monocular video. We design ExAvatar as a combination of the whole-body parametric mesh model (SMPL-X) and 3D Gaussian Splatting (3DGS). The main challenges are 1) a limited diversity of facial expressions and poses in the video and 2) the absence of 3D observations, such as 3D scans and RGBD images. The limited diversity in the video makes animations with novel facial expressions and poses non-trivial. In addition, the absence of 3D observations could cause significant ambiguity in human parts that are not observed in the video, which can result in noticeable artifacts under novel motions. To address them, we introduce our hybrid representation of the mesh and 3D Gaussians. Our hybrid representation treats each 3D Gaussian as a vertex on the surface with pre-defined connectivity information (i.e., triangle faces) between them following the mesh topology of SMPL-X. It makes our ExAvatar animatable with novel facial expressions by driven by the facial expression space of SMPL-X. In addition, by using connectivity-based regularizers, we significantly reduce artifacts in novel facial expressions and poses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20727v1">SceneTeller: Language-to-3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 ECCV'24 camera-ready version
    </div>
    <details class="paper-abstract">
      Designing high-quality indoor 3D scenes is important in many practical applications, such as room planning or game development. Conventionally, this has been a time-consuming process which requires both artistic skill and familiarity with professional software, making it hardly accessible for layman users. However, recent advances in generative AI have established solid foundation for democratizing 3D design. In this paper, we propose a pioneering approach for text-based 3D room design. Given a prompt in natural language describing the object placement in the room, our method produces a high-quality 3D scene corresponding to it. With an additional text prompt the users can change the appearance of the entire scene or of individual objects in it. Built using in-context learning, CAD model retrieval and 3D-Gaussian-Splatting-based stylization, our turnkey pipeline produces state-of-the-art 3D scenes, while being easy to use even for novices. Our project page is available at https://sceneteller.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.07504v2">COLMAP-Free 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-30
      | 💬 Project Page: https://oasisyang.github.io/colmap-free-3dgs
    </div>
    <details class="paper-abstract">
      While neural rendering has led to impressive advances in scene reconstruction and novel view synthesis, it relies heavily on accurately pre-computed camera poses. To relax this constraint, multiple efforts have been made to train Neural Radiance Fields (NeRFs) without pre-processed camera poses. However, the implicit representations of NeRFs provide extra challenges to optimize the 3D structure and camera poses at the same time. On the other hand, the recently proposed 3D Gaussian Splatting provides new opportunities given its explicit point cloud representations. This paper leverages both the explicit geometric representation and the continuity of the input video stream to perform novel view synthesis without any SfM preprocessing. We process the input frames in a sequential manner and progressively grow the 3D Gaussians set by taking one input frame at a time, without the need to pre-compute the camera poses. Our method significantly improves over previous approaches in view synthesis and camera pose estimation under large motion changes. Our project page is https://oasisyang.github.io/colmap-free-3dgs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.20055v2">SpotlessSplats: Ignoring Distractors in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a promising technique for 3D reconstruction, offering efficient training and rendering speeds, making it suitable for real-time applications.However, current methods require highly controlled environments (no moving people or wind-blown elements, and consistent lighting) to meet the inter-view consistency assumption of 3DGS. This makes reconstruction of real-world captures problematic. We present SpotLessSplats, an approach that leverages pre-trained and general-purpose features coupled with robust optimization to effectively ignore transient distractors. Our method achieves state-of-the-art reconstruction quality both visually and quantitatively, on casual captures. Additional results available at: https://spotlesssplats.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20213v1">Registering Neural 4D Gaussians for Endoscopic Surgery</a></div>
    <div class="paper-meta">
      📅 2024-07-29
    </div>
    <details class="paper-abstract">
      The recent advance in neural rendering has enabled the ability to reconstruct high-quality 4D scenes using neural networks. Although 4D neural reconstruction is popular, registration for such representations remains a challenging task, especially for dynamic scene registration in surgical planning and simulation. In this paper, we propose a novel strategy for dynamic surgical neural scene registration. We first utilize 4D Gaussian Splatting to represent the surgical scene and capture both static and dynamic scenes effectively. Then, a spatial aware feature aggregation method, Spatially Weight Cluttering (SWC) is proposed to accurately align the feature between surgical scenes, enabling precise and realistic surgical simulations. Lastly, we present a novel strategy of deformable scene registration to register two dynamic scenes. By incorporating both spatial and temporal information for correspondence matching, our approach achieves superior performance compared to existing registration methods for implicit neural representation. The proposed method has the potential to improve surgical planning and training, ultimately leading to better patient outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20194v1">Radiance Fields for Robotic Teleoperation</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 8 pages, 10 figures, Accepted to IROS 2024
    </div>
    <details class="paper-abstract">
      Radiance field methods such as Neural Radiance Fields (NeRFs) or 3D Gaussian Splatting (3DGS), have revolutionized graphics and novel view synthesis. Their ability to synthesize new viewpoints with photo-realistic quality, as well as capture complex volumetric and specular scenes, makes them an ideal visualization for robotic teleoperation setups. Direct camera teleoperation provides high-fidelity operation at the cost of maneuverability, while reconstruction-based approaches offer controllable scenes with lower fidelity. With this in mind, we propose replacing the traditional reconstruction-visualization components of the robotic teleoperation pipeline with online Radiance Fields, offering highly maneuverable scenes with photorealistic quality. As such, there are three main contributions to state of the art: (1) online training of Radiance Fields using live data from multiple cameras, (2) support for a variety of radiance methods including NeRF and 3DGS, (3) visualization suite for these methods including a virtual reality scene. To enable seamless integration with existing setups, these components were tested with multiple robots in multiple configurations and were displayed using traditional tools as well as the VR headset. The results across methods and robots were compared quantitatively to a baseline of mesh reconstruction, and a user study was conducted to compare the different visualization methods. For videos and code, check out https://leggedrobotics.github.io/rffr.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18214v2">Trimming the Fat: Efficient Compression of 3D Gaussian Splats through Pruning</a></div>
    <div class="paper-meta">
      📅 2024-07-29
      | 💬 Accepted at BMVC 2024
    </div>
    <details class="paper-abstract">
      In recent times, the utilization of 3D models has gained traction, owing to the capacity for end-to-end training initially offered by Neural Radiance Fields and more recently by 3D Gaussian Splatting (3DGS) models. The latter holds a significant advantage by inherently easing rapid convergence during training and offering extensive editability. However, despite rapid advancements, the literature still lives in its infancy regarding the scalability of these models. In this study, we take some initial steps in addressing this gap, showing an approach that enables both the memory and computational scalability of such models. Specifically, we propose "Trimming the fat", a post-hoc gradient-informed iterative pruning technique to eliminate redundant information encoded in the model. Our experimental findings on widely acknowledged benchmarks attest to the effectiveness of our approach, revealing that up to 75% of the Gaussians can be removed while maintaining or even improving upon baseline performance. Our approach achieves around 50$\times$ compression while preserving performance similar to the baseline model, and is able to speed-up computation up to 600 FPS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17596v2">GOI: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane</a></div>
    <div class="paper-meta">
      📅 2024-07-27
      | 💬 Our project page is available at https://quyans.github.io/GOI-Hyperplane/
    </div>
    <details class="paper-abstract">
      3D open-vocabulary scene understanding, crucial for advancing augmented reality and robotic applications, involves interpreting and locating specific regions within a 3D space as directed by natural language instructions. To this end, we introduce GOI, a framework that integrates semantic features from 2D vision-language foundation models into 3D Gaussian Splatting (3DGS) and identifies 3D Gaussians of Interest using an Optimizable Semantic-space Hyperplane. Our approach includes an efficient compression method that utilizes scene priors to condense noisy high-dimensional semantic features into compact low-dimensional vectors, which are subsequently embedded in 3DGS. During the open-vocabulary querying process, we adopt a distinct approach compared to existing methods, which depend on a manually set fixed empirical threshold to select regions based on their semantic feature distance to the query text embedding. This traditional approach often lacks universal accuracy, leading to challenges in precisely identifying specific target areas. Instead, our method treats the feature selection process as a hyperplane division within the feature space, retaining only those features that are highly relevant to the query. We leverage off-the-shelf 2D Referring Expression Segmentation (RES) models to fine-tune the semantic-space hyperplane, enabling a more precise distinction between target regions and others. This fine-tuning substantially improves the accuracy of open-vocabulary queries, ensuring the precise localization of pertinent 3D Gaussians. Extensive experiments demonstrate GOI's superiority over previous state-of-the-art methods. Our project page is available at https://quyans.github.io/GOI-Hyperplane/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19035v1">ScalingGaussian: Enhancing 3D Content Creation with Generative Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-26
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      The creation of high-quality 3D assets is paramount for applications in digital heritage preservation, entertainment, and robotics. Traditionally, this process necessitates skilled professionals and specialized software for the modeling, texturing, and rendering of 3D objects. However, the rising demand for 3D assets in gaming and virtual reality (VR) has led to the creation of accessible image-to-3D technologies, allowing non-professionals to produce 3D content and decreasing dependence on expert input. Existing methods for 3D content generation struggle to simultaneously achieve detailed textures and strong geometric consistency. We introduce a novel 3D content creation framework, ScalingGaussian, which combines 3D and 2D diffusion models to achieve detailed textures and geometric consistency in generated 3D assets. Initially, a 3D diffusion model generates point clouds, which are then densified through a process of selecting local regions, introducing Gaussian noise, followed by using local density-weighted selection. To refine the 3D gaussians, we utilize a 2D diffusion model with Score Distillation Sampling (SDS) loss, guiding the 3D Gaussians to clone and split. Finally, the 3D Gaussians are converted into meshes, and the surface textures are optimized using Mean Square Error(MSE) and Gradient Profile Prior(GPP) losses. Our method addresses the common issue of sparse point clouds in 3D diffusion, resulting in improved geometric structure and detailed textures. Experiments on image-to-3D tasks demonstrate that our approach efficiently generates high-quality 3D assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03613v5">Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-26
      | 💬 ECCV 2024. Project page: https://jeongminb.github.io/e-d3dgs/
    </div>
    <details class="paper-abstract">
      As 3D Gaussian Splatting (3DGS) provides fast and high-quality novel view synthesis, it is a natural extension to deform a canonical 3DGS to multiple frames for representing a dynamic scene. However, previous works fail to accurately reconstruct complex dynamic scenes. We attribute the failure to the design of the deformation field, which is built as a coordinate-based function. This approach is problematic because 3DGS is a mixture of multiple fields centered at the Gaussians, not just a single coordinate-based framework. To resolve this problem, we define the deformation as a function of per-Gaussian embeddings and temporal embeddings. Moreover, we decompose deformations as coarse and fine deformations to model slow and fast movements, respectively. Also, we introduce a local smoothness regularization for per-Gaussian embedding to improve the details in dynamic regions. Project page: https://jeongminb.github.io/e-d3dgs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18046v1">GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2024-07-25
      | 💬 13 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Implicit neural representations (INRs) have significantly advanced the field of arbitrary-scale super-resolution (ASSR) of images. Most existing INR-based ASSR networks first extract features from the given low-resolution image using an encoder, and then render the super-resolved result via a multi-layer perceptron decoder. Although these approaches have shown promising results, their performance is constrained by the limited representation ability of discrete latent codes in the encoded features. In this paper, we propose a novel ASSR method named GaussianSR that overcomes this limitation through 2D Gaussian Splatting (2DGS). Unlike traditional methods that treat pixels as discrete points, GaussianSR represents each pixel as a continuous Gaussian field. The encoded features are simultaneously refined and upsampled by rendering the mutually stacked Gaussian fields. As a result, long-range dependencies are established to enhance representation ability. In addition, a classifier is developed to dynamically assign Gaussian kernels to all pixels to further improve flexibility. All components of GaussianSR (i.e., encoder, classifier, Gaussian kernels, and decoder) are jointly learned end-to-end. Experiments demonstrate that GaussianSR achieves superior ASSR performance with fewer parameters than existing methods while enjoying interpretable and content-aware feature aggregations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06903v2">DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-25
    </div>
    <details class="paper-abstract">
      The increasing demand for virtual reality applications has highlighted the significance of crafting immersive 3D assets. We present a text-to-3D 360$^{\circ}$ scene generation pipeline that facilitates the creation of comprehensive 360$^{\circ}$ scenes for in-the-wild environments in a matter of minutes. Our approach utilizes the generative power of a 2D diffusion model and prompt self-refinement to create a high-quality and globally coherent panoramic image. This image acts as a preliminary "flat" (2D) scene representation. Subsequently, it is lifted into 3D Gaussians, employing splatting techniques to enable real-time exploration. To produce consistent 3D geometry, our pipeline constructs a spatially coherent structure by aligning the 2D monocular depth into a globally optimized point cloud. This point cloud serves as the initial state for the centroids of 3D Gaussians. In order to address invisible issues inherent in single-view inputs, we impose semantic and geometric constraints on both synthesized and input camera views as regularizations. These guide the optimization of Gaussians, aiding in the reconstruction of unseen regions. In summary, our method offers a globally consistent 3D scene within a 360$^{\circ}$ perspective, providing an enhanced immersive experience over existing techniques. Project website at: http://dreamscene360.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16037v2">GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions</a></div>
    <div class="paper-meta">
      📅 2024-07-24
      | 💬 CVPR 2024, Project page: https://GaussianEditor.github.io
    </div>
    <details class="paper-abstract">
      Recently, impressive results have been achieved in 3D scene editing with text instructions based on a 2D diffusion model. However, current diffusion models primarily generate images by predicting noise in the latent space, and the editing is usually applied to the whole image, which makes it challenging to perform delicate, especially localized, editing for 3D scenes. Inspired by recent 3D Gaussian splatting, we propose a systematic framework, named GaussianEditor, to edit 3D scenes delicately via 3D Gaussians with text instructions. Benefiting from the explicit property of 3D Gaussians, we design a series of techniques to achieve delicate editing. Specifically, we first extract the region of interest (RoI) corresponding to the text instruction, aligning it to 3D Gaussians. The Gaussian RoI is further used to control the editing process. Our framework can achieve more delicate and precise editing of 3D scenes than previous methods while enjoying much faster training speed, i.e. within 20 minutes on a single V100 GPU, more than twice as fast as Instruct-NeRF2NeRF (45 minutes -- 2 hours).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16503v1">HDRSplat: Gaussian Splatting for High Dynamic Range 3D Scene Reconstruction from Raw Images</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      The recent advent of 3D Gaussian Splatting (3DGS) has revolutionized the 3D scene reconstruction space enabling high-fidelity novel view synthesis in real-time. However, with the exception of RawNeRF, all prior 3DGS and NeRF-based methods rely on 8-bit tone-mapped Low Dynamic Range (LDR) images for scene reconstruction. Such methods struggle to achieve accurate reconstructions in scenes that require a higher dynamic range. Examples include scenes captured in nighttime or poorly lit indoor spaces having a low signal-to-noise ratio, as well as daylight scenes with shadow regions exhibiting extreme contrast. Our proposed method HDRSplat tailors 3DGS to train directly on 14-bit linear raw images in near darkness which preserves the scenes' full dynamic range and content. Our key contributions are two-fold: Firstly, we propose a linear HDR space-suited loss that effectively extracts scene information from noisy dark regions and nearly saturated bright regions simultaneously, while also handling view-dependent colors without increasing the degree of spherical harmonics. Secondly, through careful rasterization tuning, we implicitly overcome the heavy reliance and sensitivity of 3DGS on point cloud initialization. This is critical for accurate reconstruction in regions of low texture, high depth of field, and low illumination. HDRSplat is the fastest method to date that does 14-bit (HDR) 3D scene reconstruction in $\le$15 minutes/scene ($\sim$30x faster than prior state-of-the-art RawNeRF). It also boasts the fastest inference speed at $\ge$120fps. We further demonstrate the applicability of our HDR scene reconstruction by showcasing various applications like synthetic defocus, dense depth map extraction, and post-capture control of exposure, tone-mapping and view-point.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15259v3">FlowMap: High-Quality Camera Poses, Intrinsics, and Depth via Gradient Descent</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 Project website: https://cameronosmith.github.io/flowmap/
    </div>
    <details class="paper-abstract">
      This paper introduces FlowMap, an end-to-end differentiable method that solves for precise camera poses, camera intrinsics, and per-frame dense depth of a video sequence. Our method performs per-video gradient-descent minimization of a simple least-squares objective that compares the optical flow induced by depth, intrinsics, and poses against correspondences obtained via off-the-shelf optical flow and point tracking. Alongside the use of point tracks to encourage long-term geometric consistency, we introduce differentiable re-parameterizations of depth, intrinsics, and pose that are amenable to first-order optimization. We empirically show that camera parameters and dense depth recovered by our method enable photo-realistic novel view synthesis on 360-degree trajectories using Gaussian Splatting. Our method not only far outperforms prior gradient-descent based bundle adjustment methods, but surprisingly performs on par with COLMAP, the state-of-the-art SfM method, on the downstream task of 360-degree novel view synthesis (even though our method is purely gradient-descent based, fully differentiable, and presents a complete departure from conventional SfM).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.11535v3">EndoGS: Deformable Endoscopic Tissues Reconstruction with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 Accepted by Embodied AI and Robotics for HealTHcare of International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI EARTH 2024). 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Surgical 3D reconstruction is a critical area of research in robotic surgery, with recent works adopting variants of dynamic radiance fields to achieve success in 3D reconstruction of deformable tissues from single-viewpoint videos. However, these methods often suffer from time-consuming optimization or inferior quality, limiting their adoption in downstream tasks. Inspired by 3D Gaussian Splatting, a recent trending 3D representation, we present EndoGS, applying Gaussian Splatting for deformable endoscopic tissue reconstruction. Specifically, our approach incorporates deformation fields to handle dynamic scenes, depth-guided supervision with spatial-temporal weight masks to optimize 3D targets with tool occlusion from a single viewpoint, and surface-aligned regularization terms to capture the much better geometry. As a result, EndoGS reconstructs and renders high-quality deformable endoscopic tissues from a single-viewpoint video, estimated depth maps, and labeled tool masks. Experiments on DaVinci robotic surgery videos demonstrate that EndoGS achieves superior rendering quality. Code is available at https://github.com/HKU-MedAI/EndoGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16173v1">Integrating Meshes and 3D Gaussians for Indoor Scene Reconstruction with SAM Mask Guidance</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      We present a novel approach for 3D indoor scene reconstruction that combines 3D Gaussian Splatting (3DGS) with mesh representations. We use meshes for the room layout of the indoor scene, such as walls, ceilings, and floors, while employing 3D Gaussians for other objects. This hybrid approach leverages the strengths of both representations, offering enhanced flexibility and ease of editing. However, joint training of meshes and 3D Gaussians is challenging because it is not clear which primitive should affect which part of the rendered image. Objects close to the room layout often struggle during training, particularly when the room layout is textureless, which can lead to incorrect optimizations and unnecessary 3D Gaussians. To overcome these challenges, we employ Segment Anything Model (SAM) to guide the selection of primitives. The SAM mask loss enforces each instance to be represented by either Gaussians or meshes, ensuring clear separation and stable training. Furthermore, we introduce an additional densification stage without resetting the opacity after the standard densification. This stage mitigates the degradation of image quality caused by a limited number of 3D Gaussians after the standard densification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15318v2">Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering</a></div>
    <div class="paper-meta">
      📅 2024-07-23
    </div>
    <details class="paper-abstract">
      We demonstrate the feasibility of integrating physics-based animations of solids and fluids with 3D Gaussian Splatting (3DGS) to create novel effects in virtual scenes reconstructed using 3DGS. Leveraging the coherence of the Gaussian Splatting and Position-Based Dynamics (PBD) in the underlying representation, we manage rendering, view synthesis, and the dynamics of solids and fluids in a cohesive manner. Similar to GaussianShader, we enhance each Gaussian kernel with an added normal, aligning the kernel's orientation with the surface normal to refine the PBD simulation. This approach effectively eliminates spiky noises that arise from rotational deformation in solids. It also allows us to integrate physically based rendering to augment the dynamic surface reflections on fluids. Consequently, our framework is capable of realistically reproducing surface highlights on dynamic fluids and facilitating interactions between scene objects and fluids from new views. For more information, please visit our project page at \url{https://gaussiansplashing.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15484v1">6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Project page: https://mbortolon97.github.io/6dgs/ Accepted to ECCV 2024
    </div>
    <details class="paper-abstract">
      We propose 6DGS to estimate the camera pose of a target RGB image given a 3D Gaussian Splatting (3DGS) model representing the scene. 6DGS avoids the iterative process typical of analysis-by-synthesis methods (e.g. iNeRF) that also require an initialization of the camera pose in order to converge. Instead, our method estimates a 6DoF pose by inverting the 3DGS rendering process. Starting from the object surface, we define a radiant Ellicell that uniformly generates rays departing from each ellipsoid that parameterize the 3DGS model. Each Ellicell ray is associated with the rendering parameters of each ellipsoid, which in turn is used to obtain the best bindings between the target image pixels and the cast rays. These pixel-ray bindings are then ranked to select the best scoring bundle of rays, which their intersection provides the camera center and, in turn, the camera rotation. The proposed solution obviates the necessity of an "a priori" pose for initialization, and it solves 6DoF pose estimation in closed form, without the need for iterations. Moreover, compared to the existing Novel View Synthesis (NVS) baselines for pose estimation, 6DGS can improve the overall average rotational accuracy by 12% and translation accuracy by 22% on real scenes, despite not requiring any initialization pose. At the same time, our method operates near real-time, reaching 15fps on consumer hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.03890v4">A Survey on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Ongoing project
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (GS) has recently emerged as a transformative technique in the realm of explicit radiance field and computer graphics. This innovative approach, characterized by the utilization of millions of learnable 3D Gaussians, represents a significant departure from mainstream neural radiance field approaches, which predominantly use implicit, coordinate-based models to map spatial coordinates to pixel values. 3D GS, with its explicit scene representation and differentiable rendering algorithm, not only promises real-time rendering capability but also introduces unprecedented levels of editability. This positions 3D GS as a potential game-changer for the next generation of 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in the domain of 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research in this domain. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in applicable and explicit radiance field representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08742v3">Efficient4D: Fast Dynamic 3D Object Generation from a Single-view Video</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Generating dynamic 3D object from a single-view video is challenging due to the lack of 4D labeled data. An intuitive approach is to extend previous image-to-3D pipelines by transferring off-the-shelf image generation models such as score distillation sampling.However, this approach would be slow and expensive to scale due to the need for back-propagating the information-limited supervision signals through a large pretrained model. To address this, we propose an efficient video-to-4D object generation framework called Efficient4D. It generates high-quality spacetime-consistent images under different camera views, and then uses them as labeled data to directly reconstruct the 4D content through a 4D Gaussian splatting model. Importantly, our method can achieve real-time rendering under continuous camera trajectories. To enable robust reconstruction under sparse views, we introduce inconsistency-aware confidence-weighted loss design, along with a lightly weighted score distillation loss. Extensive experiments on both synthetic and real videos show that Efficient4D offers a remarkable 10-fold increase in speed when compared to prior art alternatives while preserving the quality of novel view synthesis. For example, Efficient4D takes only 10 minutes to model a dynamic object, vs 120 minutes by the previous art model Consistent4D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15187v1">HoloDreamer: Holistic 3D Panoramic World Generation from Text Descriptions</a></div>
    <div class="paper-meta">
      📅 2024-07-21
      | 💬 Homepage: https://zhouhyocean.github.io/holodreamer
    </div>
    <details class="paper-abstract">
      3D scene generation is in high demand across various domains, including virtual reality, gaming, and the film industry. Owing to the powerful generative capabilities of text-to-image diffusion models that provide reliable priors, the creation of 3D scenes using only text prompts has become viable, thereby significantly advancing researches in text-driven 3D scene generation. In order to obtain multiple-view supervision from 2D diffusion models, prevailing methods typically employ the diffusion model to generate an initial local image, followed by iteratively outpainting the local image using diffusion models to gradually generate scenes. Nevertheless, these outpainting-based approaches prone to produce global inconsistent scene generation results without high degree of completeness, restricting their broader applications. To tackle these problems, we introduce HoloDreamer, a framework that first generates high-definition panorama as a holistic initialization of the full 3D scene, then leverage 3D Gaussian Splatting (3D-GS) to quickly reconstruct the 3D scene, thereby facilitating the creation of view-consistent and fully enclosed 3D scenes. Specifically, we propose Stylized Equirectangular Panorama Generation, a pipeline that combines multiple diffusion models to enable stylized and detailed equirectangular panorama generation from complex text prompts. Subsequently, Enhanced Two-Stage Panorama Reconstruction is introduced, conducting a two-stage optimization of 3D-GS to inpaint the missing region and enhance the integrity of the scene. Comprehensive experiments demonstrated that our method outperforms prior works in terms of overall visual consistency and harmony as well as reconstruction quality and rendering robustness when generating fully enclosed scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13584v2">Connecting Consistency Distillation to Score Distillation for Text-to-3D Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 Paper accepted by ECCV2024
    </div>
    <details class="paper-abstract">
      Although recent advancements in text-to-3D generation have significantly improved generation quality, issues like limited level of detail and low fidelity still persist, which requires further improvement. To understand the essence of those issues, we thoroughly analyze current score distillation methods by connecting theories of consistency distillation to score distillation. Based on the insights acquired through analysis, we propose an optimization framework, Guided Consistency Sampling (GCS), integrated with 3D Gaussian Splatting (3DGS) to alleviate those issues. Additionally, we have observed the persistent oversaturation in the rendered views of generated 3D assets. From experiments, we find that it is caused by unwanted accumulated brightness in 3DGS during optimization. To mitigate this issue, we introduce a Brightness-Equalized Generation (BEG) scheme in 3DGS rendering. Experimental results demonstrate that our approach generates 3D assets with more details and higher fidelity than state-of-the-art methods. The codes are released at https://github.com/LMozart/ECCV2024-GCS-BEG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14846v1">Realistic Surgical Image Dataset Generation Based On 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 This paper has already been accepted by INTERNATIONAL CONFERENCE ON MEDICAL IMAGE COMPUTING AND COMPUTER ASSISTED INTERVENTION (MICCAI 2024)
    </div>
    <details class="paper-abstract">
      Computer vision technologies markedly enhance the automation capabilities of robotic-assisted minimally invasive surgery (RAMIS) through advanced tool tracking, detection, and localization. However, the limited availability of comprehensive surgical datasets for training represents a significant challenge in this field. This research introduces a novel method that employs 3D Gaussian Splatting to generate synthetic surgical datasets. We propose a method for extracting and combining 3D Gaussian representations of surgical instruments and background operating environments, transforming and combining them to generate high-fidelity synthetic surgical scenarios. We developed a data recording system capable of acquiring images alongside tool and camera poses in a surgical scene. Using this pose data, we synthetically replicate the scene, thereby enabling direct comparisons of the synthetic image quality (29.592 PSNR). As a further validation, we compared two YOLOv5 models trained on the synthetic and real data, respectively, and assessed their performance in an unseen real-world test dataset. Comparing the performances, we observe an improvement in neural network performance, with the synthetic-trained model outperforming the real-world trained model by 12%, testing both on real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14197v1">A Benchmark for Gaussian Splatting Compression and Quality Assessment Study</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      To fill the gap of traditional GS compression method, in this paper, we first propose a simple and effective GS data compression anchor called Graph-based GS Compression (GGSC). GGSC is inspired by graph signal processing theory and uses two branches to compress the primitive center and attributes. We split the whole GS sample via KDTree and clip the high-frequency components after the graph Fourier transform. Followed by quantization, G-PCC and adaptive arithmetic coding are used to compress the primitive center and attribute residual matrix to generate the bitrate file. GGSS is the first work to explore traditional GS compression, with advantages that can reveal the GS distortion characteristics corresponding to typical compression operation, such as high-frequency clipping and quantization. Second, based on GGSC, we create a GS Quality Assessment dataset (GSQA) with 120 samples. A subjective experiment is conducted in a laboratory environment to collect subjective scores after rendering GS into Processed Video Sequences (PVS). We analyze the characteristics of different GS distortions based on Mean Opinion Scores (MOS), demonstrating the sensitivity of different attributes distortion to visual quality. The GGSC code and the dataset, including GS samples, MOS, and PVS, are made publicly available at https://github.com/Qi-Yangsjtu/GGSC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14053v1">DirectL: Efficient Radiance Fields Rendering for 3D Light Field Displays</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      Autostereoscopic display, despite decades of development, has not achieved extensive application, primarily due to the daunting challenge of 3D content creation for non-specialists. The emergence of Radiance Field as an innovative 3D representation has markedly revolutionized the domains of 3D reconstruction and generation. This technology greatly simplifies 3D content creation for common users, broadening the applicability of Light Field Displays (LFDs). However, the combination of these two fields remains largely unexplored. The standard paradigm to create optimal content for parallax-based light field displays demands rendering at least 45 slightly shifted views preferably at high resolution per frame, a substantial hurdle for real-time rendering. We introduce DirectL, a novel rendering paradigm for Radiance Fields on 3D displays. We thoroughly analyze the interweaved mapping of spatial rays to screen subpixels, precisely determine the light rays entering the human eye, and propose subpixel repurposing to significantly reduce the pixel count required for rendering. Tailored for the two predominant radiance fields--Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS), we propose corresponding optimized rendering pipelines that directly render the light field images instead of multi-view images. Extensive experiments across various displays and user study demonstrate that DirectL accelerates rendering by up to 40 times compared to the standard paradigm without sacrificing visual quality. Its rendering process-only modification allows seamless integration into subsequent radiance field tasks. Finally, we integrate DirectL into diverse applications, showcasing the stunning visual experiences and the synergy between LFDs and Radiance Fields, which unveils tremendous potential for commercialization applications. \href{direct-l.github.io}{\textbf{Project Homepage}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14627v2">MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 ECCV2024, Project page: https://donydchen.github.io/mvsplat, Code: https://github.com/donydchen/mvsplat
    </div>
    <details class="paper-abstract">
      We introduce MVSplat, an efficient model that, given sparse multi-view images as input, predicts clean feed-forward 3D Gaussians. To accurately localize the Gaussian centers, we build a cost volume representation via plane sweeping, where the cross-view feature similarities stored in the cost volume can provide valuable geometry cues to the estimation of depth. We also learn other Gaussian primitives' parameters jointly with the Gaussian centers while only relying on photometric supervision. We demonstrate the importance of the cost volume representation in learning feed-forward Gaussians via extensive experimental evaluations. On the large-scale RealEstate10K and ACID benchmarks, MVSplat achieves state-of-the-art performance with the fastest feed-forward inference speed (22~fps). More impressively, compared to the latest state-of-the-art method pixelSplat, MVSplat uses $10\times$ fewer parameters and infers more than $2\times$ faster while providing higher appearance and geometry quality as well as better cross-dataset generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.13308v2">SWinGS: Sliding Windows for Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Novel view synthesis has shown rapid progress recently, with methods capable of producing increasingly photorealistic results. 3D Gaussian Splatting has emerged as a promising method, producing high-quality renderings of scenes and enabling interactive viewing at real-time frame rates. However, it is limited to static scenes. In this work, we extend 3D Gaussian Splatting to reconstruct dynamic scenes. We model a scene's dynamics using dynamic MLPs, learning deformations from temporally-local canonical representations to per-frame 3D Gaussians. To disentangle static and dynamic regions, tuneable parameters weigh each Gaussian's respective MLP parameters, improving the dynamics modelling of imbalanced scenes. We introduce a sliding window training strategy that partitions the sequence into smaller manageable windows to handle arbitrary length scenes while maintaining high rendering quality. We propose an adaptive sampling strategy to determine appropriate window size hyperparameters based on the scene's motion, balancing training overhead with visual quality. Training a separate dynamic 3D Gaussian model for each sliding window allows the canonical representation to change, enabling the reconstruction of scenes with significant geometric changes. Temporal consistency is enforced using a fine-tuning step with self-supervising consistency loss on randomly sampled novel views. As a result, our method produces high-quality renderings of general dynamic scenes with competitive quantitative performance, which can be viewed in real-time in our dynamic interactive viewer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08321v2">ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 https://guanxinglu.github.io/ManiGaussian/
    </div>
    <details class="paper-abstract">
      Performing language-conditioned robotic manipulation tasks in unstructured environments is highly demanded for general intelligent robots. Conventional robotic manipulation methods usually learn semantic representation of the observation for action prediction, which ignores the scene-level spatiotemporal dynamics for human goal completion. In this paper, we propose a dynamic Gaussian Splatting method named ManiGaussian for multi-task robotic manipulation, which mines scene dynamics via future scene reconstruction. Specifically, we first formulate the dynamic Gaussian Splatting framework that infers the semantics propagation in the Gaussian embedding space, where the semantic representation is leveraged to predict the optimal robot action. Then, we build a Gaussian world model to parameterize the distribution in our dynamic Gaussian Splatting framework, which provides informative supervision in the interactive environment via future scene reconstruction. We evaluate our ManiGaussian on 10 RLBench tasks with 166 variations, and the results demonstrate our framework can outperform the state-of-the-art methods by 13.1\% in average success rate. Project page: https://guanxinglu.github.io/ManiGaussian/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12777v1">Generalizable Human Gaussians for Sparse View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      Recent progress in neural rendering has brought forth pioneering methods, such as NeRF and Gaussian Splatting, which revolutionize view rendering across various domains like AR/VR, gaming, and content creation. While these methods excel at interpolating {\em within the training data}, the challenge of generalizing to new scenes and objects from very sparse views persists. Specifically, modeling 3D humans from sparse views presents formidable hurdles due to the inherent complexity of human geometry, resulting in inaccurate reconstructions of geometry and textures. To tackle this challenge, this paper leverages recent advancements in Gaussian Splatting and introduces a new method to learn generalizable human Gaussians that allows photorealistic and accurate view-rendering of a new human subject from a limited set of sparse views in a feed-forward manner. A pivotal innovation of our approach involves reformulating the learning of 3D Gaussian parameters into a regression process defined on the 2D UV space of a human template, which allows leveraging the strong geometry prior and the advantages of 2D convolutions. In addition, a multi-scaffold is proposed to effectively represent the offset details. Our method outperforms recent methods on both within-dataset generalization as well as cross-dataset generalization settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01810v2">GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 ECCV2024. Project Page: https://gs2mesh.github.io/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has emerged as an efficient approach for accurately representing scenes. However, despite its superior novel view synthesis capabilities, extracting the geometry of the scene directly from the Gaussian properties remains a challenge, as those are optimized based on a photometric loss. While some concurrent models have tried adding geometric constraints during the Gaussian optimization process, they still produce noisy, unrealistic surfaces. We propose a novel approach for bridging the gap between the noisy 3DGS representation and the smooth 3D mesh representation, by injecting real-world knowledge into the depth extraction process. Instead of extracting the geometry of the scene directly from the Gaussian properties, we instead extract the geometry through a pre-trained stereo-matching model. We render stereo-aligned pairs of images corresponding to the original training poses, feed the pairs into a stereo model to get a depth profile, and finally fuse all of the profiles together to get a single mesh. The resulting reconstruction is smoother, more accurate and shows more intricate details compared to other methods for surface reconstruction from Gaussian Splatting, while only requiring a small overhead on top of the fairly short 3DGS optimization process. We performed extensive testing of the proposed method on in-the-wild scenes, obtained using a smartphone, showcasing its superior reconstruction abilities. Additionally, we tested the method on the Tanks and Temples and DTU benchmarks, achieving state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11324v2">GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 accepted to ECCV 2024
    </div>
    <details class="paper-abstract">
      During the Gaussian Splatting optimization process, the scene's geometry can gradually deteriorate if its structure is not deliberately preserved, especially in non-textured regions such as walls, ceilings, and furniture surfaces. This degradation significantly affects the rendering quality of novel views that deviate significantly from the viewpoints in the training data. To mitigate this issue, we propose a novel approach called GeoGaussian. Based on the smoothly connected areas observed from point clouds, this method introduces a novel pipeline to initialize thin Gaussians aligned with the surfaces, where the characteristic can be transferred to new generations through a carefully designed densification strategy. Finally, the pipeline ensures that the scene's geometry and texture are maintained through constrained optimization processes with explicit geometry constraints. Benefiting from the proposed architecture, the generative ability of 3D Gaussians is enhanced, especially in structured regions. Our proposed pipeline achieves state-of-the-art performance in novel view synthesis and geometric reconstruction, as evaluated qualitatively and quantitatively on public datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13327v3">Gaussian Splatting on the Move: Blur and Rolling Shutter Compensation for Natural Camera Motion</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Source code available at https://github.com/SpectacularAI/3dgs-deblur
    </div>
    <details class="paper-abstract">
      High-quality scene reconstruction and novel view synthesis based on Gaussian Splatting (3DGS) typically require steady, high-quality photographs, often impractical to capture with handheld cameras. We present a method that adapts to camera motion and allows high-quality scene reconstruction with handheld video data suffering from motion blur and rolling shutter distortion. Our approach is based on detailed modelling of the physical image formation process and utilizes velocities estimated using visual-inertial odometry (VIO). Camera poses are considered non-static during the exposure time of a single image frame and camera poses are further optimized in the reconstruction process. We formulate a differentiable rendering pipeline that leverages screen space approximation to efficiently incorporate rolling-shutter and motion blur effects into the 3DGS framework. Our results with both synthetic and real data demonstrate superior performance in mitigating camera motion over existing methods, thereby advancing 3DGS in naturalistic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01133v3">CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Accepted by ECCV2024; Project Page: https://dekuliutesla.github.io/citygs/
    </div>
    <details class="paper-abstract">
      The advancement of real-time 3D scene reconstruction and novel view synthesis has been significantly propelled by 3D Gaussian Splatting (3DGS). However, effectively training large-scale 3DGS and rendering it in real-time across various scales remains challenging. This paper introduces CityGaussian (CityGS), which employs a novel divide-and-conquer training approach and Level-of-Detail (LoD) strategy for efficient large-scale 3DGS training and rendering. Specifically, the global scene prior and adaptive training data selection enables efficient training and seamless fusion. Based on fused Gaussian primitives, we generate different detail levels through compression, and realize fast rendering across various scales through the proposed block-wise detail levels selection and aggregation strategy. Extensive experimental results on large-scale scenes demonstrate that our approach attains state-of-theart rendering quality, enabling consistent real-time rendering of largescale scenes across vastly different scales. Our project page is available at https://dekuliutesla.github.io/citygs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07284v2">MIGS: Multi-Identity Gaussian Splatting via Tensor Decomposition</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Accepted by ECCV 2024. Project page: https://aggelinacha.github.io/MIGS/
    </div>
    <details class="paper-abstract">
      We introduce MIGS (Multi-Identity Gaussian Splatting), a novel method that learns a single neural representation for multiple identities, using only monocular videos. Recent 3D Gaussian Splatting (3DGS) approaches for human avatars require per-identity optimization. However, learning a multi-identity representation presents advantages in robustly animating humans under arbitrary poses. We propose to construct a high-order tensor that combines all the learnable 3DGS parameters for all the training identities. By assuming a low-rank structure and factorizing the tensor, we model the complex rigid and non-rigid deformations of multiple subjects in a unified network, significantly reducing the total number of parameters. Our proposed approach leverages information from all the training identities and enables robust animation under challenging unseen poses, outperforming existing approaches. It can also be extended to learn unseen identities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10318v2">RecGS: Removing Water Caustic with Recurrent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 8 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Water caustics are commonly observed in seafloor imaging data from shallow-water areas. Traditional methods that remove caustic patterns from images often rely on 2D filtering or pre-training on an annotated dataset, hindering the performance when generalizing to real-world seafloor data with 3D structures. In this paper, we present a novel method Recurrent Gaussian Splatting (RecGS), which takes advantage of today's photorealistic 3D reconstruction technology, 3DGS, to separate caustics from seafloor imagery. With a sequence of images taken by an underwater robot, we build 3DGS recurrently and decompose the caustic with low-pass filtering in each iteration. In the experiments, we analyze and compare with different methods, including joint optimization, 2D filtering, and deep learning approaches. The results show that our method can effectively separate the caustic from the seafloor, improving the visual appearance, and can be potentially applied on more problems with inconsistent illumination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12957v2">GVGEN: Text-to-3D Generation with Volumetric Representation</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 ECCV 2024
    </div>
    <details class="paper-abstract">
      In recent years, 3D Gaussian splatting has emerged as a powerful technique for 3D reconstruction and generation, known for its fast and high-quality rendering capabilities. To address these shortcomings, this paper introduces a novel diffusion-based framework, GVGEN, designed to efficiently generate 3D Gaussian representations from text input. We propose two innovative techniques:(1) Structured Volumetric Representation. We first arrange disorganized 3D Gaussian points as a structured form GaussianVolume. This transformation allows the capture of intricate texture details within a volume composed of a fixed number of Gaussians. To better optimize the representation of these details, we propose a unique pruning and densifying method named the Candidate Pool Strategy, enhancing detail fidelity through selective optimization. (2) Coarse-to-fine Generation Pipeline. To simplify the generation of GaussianVolume and empower the model to generate instances with detailed 3D geometry, we propose a coarse-to-fine pipeline. It initially constructs a basic geometric structure, followed by the prediction of complete Gaussian attributes. Our framework, GVGEN, demonstrates superior performance in qualitative and quantitative assessments compared to existing 3D generation methods. Simultaneously, it maintains a fast generation speed ($\sim$7 seconds), effectively striking a balance between quality and efficiency. Our project page is: https://gvgen.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11840v1">MVG-Splatting: Multi-View Guided Gaussian Splatting with Adaptive Quantile-Based Geometric Consistency Densification</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 https://mvgsplatting.github.io
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of 3D reconstruction, 3D Gaussian Splatting (3DGS) and 2D Gaussian Splatting (2DGS) represent significant advancements. Although 2DGS compresses 3D Gaussian primitives into 2D Gaussian surfels to effectively enhance mesh extraction quality, this compression can potentially lead to a decrease in rendering quality. Additionally, unreliable densification processes and the calculation of depth through the accumulation of opacity can compromise the detail of mesh extraction. To address this issue, we introduce MVG-Splatting, a solution guided by Multi-View considerations. Specifically, we integrate an optimized method for calculating normals, which, combined with image gradients, helps rectify inconsistencies in the original depth computations. Additionally, utilizing projection strategies akin to those in Multi-View Stereo (MVS), we propose an adaptive quantile-based method that dynamically determines the level of additional densification guided by depth maps, from coarse to fine detail. Experimental evidence demonstrates that our method not only resolves the issues of rendering quality degradation caused by depth discrepancies but also facilitates direct mesh extraction from dense Gaussian point clouds using the Marching Cubes algorithm. This approach significantly enhances the overall fidelity and accuracy of the 3D reconstruction process, ensuring that both the geometric details and visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11793v1">Click-Gaussian: Interactive Segmentation to Any 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 Accepted to ECCV 2024. The first two authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Interactive segmentation of 3D Gaussians opens a great opportunity for real-time manipulation of 3D scenes thanks to the real-time rendering capability of 3D Gaussian Splatting. However, the current methods suffer from time-consuming post-processing to deal with noisy segmentation output. Also, they struggle to provide detailed segmentation, which is important for fine-grained manipulation of 3D scenes. In this study, we propose Click-Gaussian, which learns distinguishable feature fields of two-level granularity, facilitating segmentation without time-consuming post-processing. We delve into challenges stemming from inconsistently learned feature fields resulting from 2D segmentation obtained independently from a 3D scene. 3D segmentation accuracy deteriorates when 2D segmentation results across the views, primary cues for 3D segmentation, are in conflict. To overcome these issues, we propose Global Feature-guided Learning (GFL). GFL constructs the clusters of global feature candidates from noisy 2D segments across the views, which smooths out noises when training the features of 3D Gaussians. Our method runs in 10 ms per click, 15 to 130 times as fast as the previous methods, while also significantly improving segmentation accuracy. Our project page is available at https://seokhunchoi.github.io/Click-Gaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00609v4">SuperGaussian: Repurposing Video Models for 3D Super Resolution</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 Accepted at ECCV 2024, project website with interactive demo: https://supergaussian.github.io
    </div>
    <details class="paper-abstract">
      We present a simple, modular, and generic method that upsamples coarse 3D models by adding geometric and appearance details. While generative 3D models now exist, they do not yet match the quality of their counterparts in image and video domains. We demonstrate that it is possible to directly repurpose existing (pretrained) video models for 3D super-resolution and thus sidestep the problem of the shortage of large repositories of high-quality 3D training models. We describe how to repurpose video upsampling models, which are not 3D consistent, and combine them with 3D consolidation to produce 3D-consistent results. As output, we produce high quality Gaussian Splat models, which are object centric and effective. Our method is category agnostic and can be easily incorporated into existing 3D workflows. We evaluate our proposed SuperGaussian on a variety of 3D inputs, which are diverse both in terms of complexity and representation (e.g., Gaussian Splats or NeRFs), and demonstrate that our simple method significantly improves the fidelity of the final 3D models. Check our project website for details: supergaussian.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11347v1">I$^2$-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 ECCV 2024
    </div>
    <details class="paper-abstract">
      We present an inverse image-formation module that can enhance the robustness of existing visual SLAM pipelines for casually captured scenarios. Casual video captures often suffer from motion blur and varying appearances, which degrade the final quality of coherent 3D visual representation. We propose integrating the physical imaging into the SLAM system, which employs linear HDR radiance maps to collect measurements. Specifically, individual frames aggregate images of multiple poses along the camera trajectory to explain prevalent motion blur in hand-held videos. Additionally, we accommodate per-frame appearance variation by dedicating explicit variables for image formation steps, namely white balance, exposure time, and camera response function. Through joint optimization of additional variables, the SLAM pipeline produces high-quality images with more accurate trajectories. Extensive experiments demonstrate that our approach can be incorporated into recent visual SLAM pipelines using various scene representations, such as neural radiance fields or Gaussian splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11343v1">Ev-GS: Event-based Gaussian splatting for Efficient and Accurate Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      Computational neuromorphic imaging (CNI) with event cameras offers advantages such as minimal motion blur and enhanced dynamic range, compared to conventional frame-based methods. Existing event-based radiance field rendering methods are built on neural radiance field, which is computationally heavy and slow in reconstruction speed. Motivated by the two aspects, we introduce Ev-GS, the first CNI-informed scheme to infer 3D Gaussian splatting from a monocular event camera, enabling efficient novel view synthesis. Leveraging 3D Gaussians with pure event-based supervision, Ev-GS overcomes challenges such as the detection of fast-moving objects and insufficient lighting. Experimental results show that Ev-GS outperforms the method that takes frame-based signals as input by rendering realistic views with reduced blurring and improved visual quality. Moreover, it demonstrates competitive reconstruction quality and reduced computing occupancy compared to existing methods, which paves the way to a highly efficient CNI approach for signal processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11309v1">Gaussian Splatting LK</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 15 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from 2D images and generating diverse views over time presents a significant challenge due to the inherent complexity and temporal dynamics involved. While recent advancements in neural implicit models and dynamic Gaussian Splatting have shown promise, limitations persist, particularly in accurately capturing the underlying geometry of highly dynamic scenes. Some approaches address this by incorporating strong semantic and geometric priors through diffusion models. However, we explore a different avenue by investigating the potential of regularizing the native warp field within the dynamic Gaussian Splatting framework. Our method is grounded on the key intuition that an accurate warp field should produce continuous space-time motions. While enforcing the motion constraints on warp fields is non-trivial, we show that we can exploit knowledge innate to the forward warp field network to derive an analytical velocity field, then time integrate for scene flows to effectively constrain both the 2D motion and 3D positions of the Gaussians. This derived Lucas-Kanade style analytical regularization enables our method to achieve superior performance in reconstructing highly dynamic scenes, even under minimal camera movement, extending the boundaries of what existing dynamic Gaussian Splatting frameworks can achieve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11174v1">iHuman: Instant Animatable Digital Humans From Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 15 pages, eccv, 2024
    </div>
    <details class="paper-abstract">
      Personalized 3D avatars require an animatable representation of digital humans. Doing so instantly from monocular videos offers scalability to broad class of users and wide-scale applications. In this paper, we present a fast, simple, yet effective method for creating animatable 3D digital humans from monocular videos. Our method utilizes the efficiency of Gaussian splatting to model both 3D geometry and appearance. However, we observed that naively optimizing Gaussian splats results in inaccurate geometry, thereby leading to poor animations. This work achieves and illustrates the need of accurate 3D mesh-type modelling of the human body for animatable digitization through Gaussian splats. This is achieved by developing a novel pipeline that benefits from three key aspects: (a) implicit modelling of surface's displacements and the color's spherical harmonics; (b) binding of 3D Gaussians to the respective triangular faces of the body template; (c) a novel technique to render normals followed by their auxiliary supervision. Our exhaustive experiments on three different benchmark datasets demonstrates the state-of-the-art results of our method, in limited time settings. In fact, our method is faster by an order of magnitude (in terms of training time) than its closest competitor. At the same time, we achieve superior rendering and 3D reconstruction performance under the change of poses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00440v3">Topo4D: Topology-Preserving Gaussian Splatting for High-Fidelity 4D Head Capture</a></div>
    <div class="paper-meta">
      📅 2024-07-15
    </div>
    <details class="paper-abstract">
      4D head capture aims to generate dynamic topological meshes and corresponding texture maps from videos, which is widely utilized in movies and games for its ability to simulate facial muscle movements and recover dynamic textures in pore-squeezing. The industry often adopts the method involving multi-view stereo and non-rigid alignment. However, this approach is prone to errors and heavily reliant on time-consuming manual processing by artists. To simplify this process, we propose Topo4D, a novel framework for automatic geometry and texture generation, which optimizes densely aligned 4D heads and 8K texture maps directly from calibrated multi-view time-series images. Specifically, we first represent the time-series faces as a set of dynamic 3D Gaussians with fixed topology in which the Gaussian centers are bound to the mesh vertices. Afterward, we perform alternative geometry and texture optimization frame-by-frame for high-quality geometry and texture learning while maintaining temporal topology stability. Finally, we can extract dynamic facial meshes in regular wiring arrangement and high-fidelity textures with pore-level details from the learned Gaussians. Extensive experiments show that our method achieves superior results than the current SOTA face reconstruction methods both in the quality of meshes and textures. Project page: https://xuanchenli.github.io/Topo4D/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10743v1">Scaling 3D Reasoning with LMMs to Large Robot Mission Environments Using Datagraphs</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 Accepted to the RSS Workshop on Semantics for Robotics: From Environment Understanding and Reasoning to Safe Interaction 2024
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of scaling Large Multimodal Models (LMMs) to expansive 3D environments. Solving this open problem is especially relevant for robot deployment in many first-responder scenarios, such as search-and-rescue missions that cover vast spaces. The use of LMMs in these settings is currently hampered by the strict context windows that limit the LMM's input size. We therefore introduce a novel approach that utilizes a datagraph structure, which allows the LMM to iteratively query smaller sections of a large environment. Using the datagraph in conjunction with graph traversal algorithms, we can prioritize the most relevant locations to the query, thereby improving the scalability of 3D scene language tasks. We illustrate the datagraph using 3D scenes, but these can be easily substituted by other dense modalities that represent the environment, such as pointclouds or Gaussian splats. We demonstrate the potential to use the datagraph for two 3D scene language task use cases, in a search-and-rescue mission example.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10707v1">Interactive Rendering of Relightable and Animatable Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2024-07-15
    </div>
    <details class="paper-abstract">
      Creating relightable and animatable avatars from multi-view or monocular videos is a challenging task for digital human creation and virtual reality applications. Previous methods rely on neural radiance fields or ray tracing, resulting in slow training and rendering processes. By utilizing Gaussian Splatting, we propose a simple and efficient method to decouple body materials and lighting from sparse-view or monocular avatar videos, so that the avatar can be rendered simultaneously under novel viewpoints, poses, and lightings at interactive frame rates (6.9 fps). Specifically, we first obtain the canonical body mesh using a signed distance function and assign attributes to each mesh vertex. The Gaussians in the canonical space then interpolate from nearby body mesh vertices to obtain the attributes. We subsequently deform the Gaussians to the posed space using forward skinning, and combine the learnable environment light with the Gaussian attributes for shading computation. To achieve fast shadow modeling, we rasterize the posed body mesh from dense viewpoints to obtain the visibility. Our approach is not only simple but also fast enough to allow interactive rendering of avatar animation under environmental light changes. Experiments demonstrate that, compared to previous works, our method can render higher quality results at a faster speed on both synthetic and real datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08528v3">4D Gaussian Splatting for Real-Time Dynamic Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 CVPR 2024. Project page: https://guanjunwu.github.io/4dgs/
    </div>
    <details class="paper-abstract">
      Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. In 4D-GS, a novel explicit representation containing both 3D Gaussians and 4D neural voxels is proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is proposed to efficiently build Gaussian features from 4D neural voxels and then a lightweight MLP is applied to predict Gaussian deformations at novel timestamps. Our 4D-GS method achieves real-time rendering under high resolutions, 82 FPS at an 800$\times$800 resolution on an RTX 3090 GPU while maintaining comparable or better quality than previous state-of-the-art methods. More demos and code are available at https://guanjunwu.github.io/4dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12218v3">MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 ECCV2024, Project page: https://mvsgaussian.github.io/ , Code: https://github.com/TQTQliu/MVSGaussian
    </div>
    <details class="paper-abstract">
      We present MVSGaussian, a new generalizable 3D Gaussian representation approach derived from Multi-View Stereo (MVS) that can efficiently reconstruct unseen scenes. Specifically, 1) we leverage MVS to encode geometry-aware Gaussian representations and decode them into Gaussian parameters. 2) To further enhance performance, we propose a hybrid Gaussian rendering that integrates an efficient volume rendering design for novel view synthesis. 3) To support fast fine-tuning for specific scenes, we introduce a multi-view geometric consistent aggregation strategy to effectively aggregate the point clouds generated by the generalizable model, serving as the initialization for per-scene optimization. Compared with previous generalizable NeRF-based methods, which typically require minutes of fine-tuning and seconds of rendering per image, MVSGaussian achieves real-time rendering with better synthesis quality for each scene. Compared with the vanilla 3D-GS, MVSGaussian achieves better view synthesis with less training computational cost. Extensive experiments on DTU, Real Forward-facing, NeRF Synthetic, and Tanks and Temples datasets validate that MVSGaussian attains state-of-the-art performance with convincing generalizability, real-time rendering speed, and fast per-scene optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.02281v2">PEGASUS: Physically Enhanced Gaussian Splatting Simulation System for 6DoF Object Pose Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 Project Page: https://meyerls.github.io/pegasus_web
    </div>
    <details class="paper-abstract">
      We introduce Physically Enhanced Gaussian Splatting Simulation System (PEGASUS) for 6DOF object pose dataset generation, a versatile dataset generator based on 3D Gaussian Splatting. Environment and object representations can be easily obtained using commodity cameras to reconstruct with Gaussian Splatting. <i>PEGASUS</i> allows the composition of new scenes by merging the respective underlying Gaussian Splatting point cloud of an environment with one or multiple objects. Leveraging a physics engine enables the simulation of natural object placement within a scene through interaction between meshes extracted for the objects and the environment. Consequently, an extensive amount of new scenes - static or dynamic - can be created by combining different environments and objects. By rendering scenes from various perspectives, diverse data points such as RGB images, depth maps, semantic masks, and 6DoF object poses can be extracted. Our study demonstrates that training on data generated by PEGASUS enables pose estimation networks to successfully transfer from synthetic data to real-world data. Moreover, we introduce the Ramen dataset, comprising 30 Japanese cup noodle items. This dataset includes spherical scans that captures images from both object hemisphere and the Gaussian Splatting reconstruction, making them compatible with PEGASUS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.15171v4">Deceptive-NeRF/3DGS: Diffusion-Generated Pseudo-Observations for High-Quality Sparse-View Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 Paper accepted to ECCV 2024. Project page: https://xinhangliu.com/deceptive-nerf-3dgs
    </div>
    <details class="paper-abstract">
      Novel view synthesis via Neural Radiance Fields (NeRFs) or 3D Gaussian Splatting (3DGS) typically necessitates dense observations with hundreds of input images to circumvent artifacts. We introduce Deceptive-NeRF/3DGS to enhance sparse-view reconstruction with only a limited set of input images, by leveraging a diffusion model pre-trained from multiview datasets. Different from using diffusion priors to regularize representation optimization, our method directly uses diffusion-generated images to train NeRF/3DGS as if they were real input views. Specifically, we propose a deceptive diffusion model turning noisy images rendered from few-view reconstructions into high-quality photorealistic pseudo-observations. To resolve consistency among pseudo-observations and real input views, we develop an uncertainty measure to guide the diffusion model's generation. Our system progressively incorporates diffusion-generated pseudo-observations into the training image sets, ultimately densifying the sparse input observations by 5 to 10 times. Extensive experiments across diverse and challenging datasets validate that our approach outperforms existing state-of-the-art methods and is capable of synthesizing novel views with super-resolution in the few-view setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08733v4">GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing</a></div>
    <div class="paper-meta">
      📅 2024-07-14
      | 💬 ECCV2024, Project Website: https://gaussctrl.active.vision/
    </div>
    <details class="paper-abstract">
      We propose GaussCtrl, a text-driven method to edit a 3D scene reconstructed by the 3D Gaussian Splatting (3DGS). Our method first renders a collection of images by using the 3DGS and edits them by using a pre-trained 2D diffusion model (ControlNet) based on the input prompt, which is then used to optimise the 3D model. Our key contribution is multi-view consistent editing, which enables editing all images together instead of iteratively editing one image while updating the 3D model as in previous works. It leads to faster editing as well as higher visual quality. This is achieved by the two terms: (a) depth-conditioned editing that enforces geometric consistency across multi-view images by leveraging naturally consistent depth maps. (b) attention-based latent code alignment that unifies the appearance of edited images by conditioning their editing to several reference views through self and cross-view attention between images' latent representations. Experiments demonstrate that our method achieves faster editing and better visual results than previous state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10102v1">3DEgo: 3D Editing on the Go!</a></div>
    <div class="paper-meta">
      📅 2024-07-14
      | 💬 ECCV 2024 Accepted Paper
    </div>
    <details class="paper-abstract">
      We introduce 3DEgo to address a novel problem of directly synthesizing photorealistic 3D scenes from monocular videos guided by textual prompts. Conventional methods construct a text-conditioned 3D scene through a three-stage process, involving pose estimation using Structure-from-Motion (SfM) libraries like COLMAP, initializing the 3D model with unedited images, and iteratively updating the dataset with edited images to achieve a 3D scene with text fidelity. Our framework streamlines the conventional multi-stage 3D editing process into a single-stage workflow by overcoming the reliance on COLMAP and eliminating the cost of model initialization. We apply a diffusion model to edit video frames prior to 3D scene creation by incorporating our designed noise blender module for enhancing multi-view editing consistency, a step that does not require additional training or fine-tuning of T2I diffusion models. 3DEgo utilizes 3D Gaussian Splatting to create 3D scenes from the multi-view consistent edited frames, capitalizing on the inherent temporal continuity and explicit point cloud data. 3DEgo demonstrates remarkable editing precision, speed, and adaptability across a variety of video sources, as validated by extensive evaluations on six datasets, including our own prepared GS25 dataset. Project Page: https://3dego.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15704v2">Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections</a></div>
    <div class="paper-meta">
      📅 2024-07-14
    </div>
    <details class="paper-abstract">
      Novel view synthesis from unconstrained in-the-wild images remains a meaningful but challenging task. The photometric variation and transient occluders in those unconstrained images make it difficult to reconstruct the original scene accurately. Previous approaches tackle the problem by introducing a global appearance feature in Neural Radiance Fields (NeRF). However, in the real world, the unique appearance of each tiny point in a scene is determined by its independent intrinsic material attributes and the varying environmental impacts it receives. Inspired by this fact, we propose Gaussian in the wild (GS-W), a method that uses 3D Gaussian points to reconstruct the scene and introduces separated intrinsic and dynamic appearance feature for each point, capturing the unchanged scene appearance along with dynamic variation like illumination and weather. Additionally, an adaptive sampling strategy is presented to allow each Gaussian point to focus on the local and detailed information more effectively. We also reduce the impact of transient occluders using a 2D visibility map. More experiments have demonstrated better reconstruction quality and details of GS-W compared to NeRF-based methods, with a faster rendering speed. Video results and code are available at https://eastbeanzhang.github.io/GS-W/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10062v1">SpikeGS: 3D Gaussian Splatting from Spike Streams with High-Speed Camera Motion</a></div>
    <div class="paper-meta">
      📅 2024-07-14
    </div>
    <details class="paper-abstract">
      Novel View Synthesis plays a crucial role by generating new 2D renderings from multi-view images of 3D scenes. However, capturing high-speed scenes with conventional cameras often leads to motion blur, hindering the effectiveness of 3D reconstruction. To address this challenge, high-frame-rate dense 3D reconstruction emerges as a vital technique, enabling detailed and accurate modeling of real-world objects or scenes in various fields, including Virtual Reality or embodied AI. Spike cameras, a novel type of neuromorphic sensor, continuously record scenes with an ultra-high temporal resolution, showing potential for accurate 3D reconstruction. Despite their promise, existing approaches, such as applying Neural Radiance Fields (NeRF) to spike cameras, encounter challenges due to the time-consuming rendering process. To address this issue, we make the first attempt to introduce the 3D Gaussian Splatting (3DGS) into spike cameras in high-speed capture, providing 3DGS as dense and continuous clues of views, then constructing SpikeGS. Specifically, to train SpikeGS, we establish computational equations between the rendering process of 3DGS and the processes of instantaneous imaging and exposing-like imaging of the continuous spike stream. Besides, we build a very lightweight but effective mapping process from spikes to instant images to support training. Furthermore, we introduced a new spike-based 3D rendering dataset for validation. Extensive experiments have demonstrated our method possesses the high quality of novel view rendering, proving the tremendous potential of spike cameras in modeling 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02945v3">VEGS: View Extrapolation of Urban Scenes in 3D Gaussian Splatting using Learned Priors</a></div>
    <div class="paper-meta">
      📅 2024-07-13
      | 💬 The first two authors contributed equally. Project Page: https://vegs3d.github.io/
    </div>
    <details class="paper-abstract">
      Neural rendering-based urban scene reconstruction methods commonly rely on images collected from driving vehicles with cameras facing and moving forward. Although these methods can successfully synthesize from views similar to training camera trajectory, directing the novel view outside the training camera distribution does not guarantee on-par performance. In this paper, we tackle the Extrapolated View Synthesis (EVS) problem by evaluating the reconstructions on views such as looking left, right or downwards with respect to training camera distributions. To improve rendering quality for EVS, we initialize our model by constructing dense LiDAR map, and propose to leverage prior scene knowledge such as surface normal estimator and large-scale diffusion model. Qualitative and quantitative comparisons demonstrate the effectiveness of our methods on EVS. To the best of our knowledge, we are the first to address the EVS problem in urban scene reconstruction. Link to our project page: https://vegs3d.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02410v2">TCLC-GS: Tightly Coupled LiDAR-Camera Gaussian Splatting for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-07-12
    </div>
    <details class="paper-abstract">
      Most 3D Gaussian Splatting (3D-GS) based methods for urban scenes initialize 3D Gaussians directly with 3D LiDAR points, which not only underutilizes LiDAR data capabilities but also overlooks the potential advantages of fusing LiDAR with camera data. In this paper, we design a novel tightly coupled LiDAR-Camera Gaussian Splatting (TCLC-GS) to fully leverage the combined strengths of both LiDAR and camera sensors, enabling rapid, high-quality 3D reconstruction and novel view RGB/depth synthesis. TCLC-GS designs a hybrid explicit (colorized 3D mesh) and implicit (hierarchical octree feature) 3D representation derived from LiDAR-camera data, to enrich the properties of 3D Gaussians for splatting. 3D Gaussian's properties are not only initialized in alignment with the 3D mesh which provides more completed 3D shape and color information, but are also endowed with broader contextual information through retrieved octree implicit features. During the Gaussian Splatting optimization process, the 3D mesh offers dense depth information as supervision, which enhances the training process by learning of a robust geometry. Comprehensive evaluations conducted on the Waymo Open Dataset and nuScenes Dataset validate our method's state-of-the-art (SOTA) performance. Utilizing a single NVIDIA RTX 3090 Ti, our method demonstrates fast training and achieves real-time RGB and depth rendering at 90 FPS in resolution of 1920x1280 (Waymo), and 120 FPS in resolution of 1600x900 (nuScenes) in urban scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09473v1">StyleSplat: 3D Object Style Transfer with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 for code and results, see http://bernard0047.github.io/stylesplat
    </div>
    <details class="paper-abstract">
      Recent advancements in radiance fields have opened new avenues for creating high-quality 3D assets and scenes. Style transfer can enhance these 3D assets with diverse artistic styles, transforming creative expression. However, existing techniques are often slow or unable to localize style transfer to specific objects. We introduce StyleSplat, a lightweight method for stylizing 3D objects in scenes represented by 3D Gaussians from reference style images. Our approach first learns a photorealistic representation of the scene using 3D Gaussian splatting while jointly segmenting individual 3D objects. We then use a nearest-neighbor feature matching loss to finetune the Gaussians of the selected objects, aligning their spherical harmonic coefficients with the style image to ensure consistency and visual appeal. StyleSplat allows for quick, customizable style transfer and localized stylization of multiple objects within a scene, each with a different style. We demonstrate its effectiveness across various 3D scenes and styles, showcasing enhanced control and customization in 3D creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04504v2">Segment Any 4D Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Modeling, understanding, and reconstructing the real world are crucial in XR/VR. Recently, 3D Gaussian Splatting (3D-GS) methods have shown remarkable success in modeling and understanding 3D scenes. Similarly, various 4D representations have demonstrated the ability to capture the dynamics of the 4D world. However, there is a dearth of research focusing on segmentation within 4D representations. In this paper, we propose Segment Any 4D Gaussians (SA4D), one of the first frameworks to segment anything in the 4D digital world based on 4D Gaussians. In SA4D, an efficient temporal identity feature field is introduced to handle Gaussian drifting, with the potential to learn precise identity features from noisy and sparse input. Additionally, a 4D segmentation refinement process is proposed to remove artifacts. Our SA4D achieves precise, high-quality segmentation within seconds in 4D Gaussians and shows the ability to remove, recolor, compose, and render high-quality anything masks. More demos are available at: https://jsxzs.github.io/sa4d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14530v3">HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression</a></div>
    <div class="paper-meta">
      📅 2024-07-12
      | 💬 Project Page: https://yihangchen-ee.github.io/project_hac/ Code: https://github.com/YihangChen-ee/HAC
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising framework for novel view synthesis, boasting rapid rendering speed with high fidelity. However, the substantial Gaussians and their associated attributes necessitate effective compression techniques. Nevertheless, the sparse and unorganized nature of the point cloud of Gaussians (or anchors in our paper) presents challenges for compression. To address this, we make use of the relations between the unorganized anchors and the structured hash grid, leveraging their mutual information for context modeling, and propose a Hash-grid Assisted Context (HAC) framework for highly compact 3DGS representation. Our approach introduces a binary hash grid to establish continuous spatial consistencies, allowing us to unveil the inherent spatial relations of anchors through a carefully designed context model. To facilitate entropy coding, we utilize Gaussian distributions to accurately estimate the probability of each quantized attribute, where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. Additionally, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors. Importantly, our work is the pioneer to explore context-based compression for 3DGS representation, resulting in a remarkable size reduction of over $75\times$ compared to vanilla 3DGS, while simultaneously improving fidelity, and achieving over $11\times$ size reduction over SOTA 3DGS compression approach Scaffold-GS. Our code is available here: https://github.com/YihangChen-ee/HAC
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12110v2">CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization</a></div>
    <div class="paper-meta">
      📅 2024-07-11
      | 💬 Accepted at ECCV 2024. Project page: https://jiaw-z.github.io/CoR-GS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) creates a radiance field consisting of 3D Gaussians to represent a scene. With sparse training views, 3DGS easily suffers from overfitting, negatively impacting rendering. This paper introduces a new co-regularization perspective for improving sparse-view 3DGS. When training two 3D Gaussian radiance fields, we observe that the two radiance fields exhibit point disagreement and rendering disagreement that can unsupervisedly predict reconstruction quality, stemming from the randomness of densification implementation. We further quantify the two disagreements and demonstrate the negative correlation between them and accurate reconstruction, which allows us to identify inaccurate reconstruction without accessing ground-truth information. Based on the study, we propose CoR-GS, which identifies and suppresses inaccurate reconstruction based on the two disagreements: (1) Co-pruning considers Gaussians that exhibit high point disagreement in inaccurate positions and prunes them. (2) Pseudo-view co-regularization considers pixels that exhibit high rendering disagreement are inaccurate and suppress the disagreement. Results on LLFF, Mip-NeRF360, DTU, and Blender demonstrate that CoR-GS effectively regularizes the scene geometry, reconstructs the compact representations, and achieves state-of-the-art novel view synthesis quality under sparse training views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01042v2">Self-Calibrating 4D Novel View Synthesis from Monocular Videos Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-11
      | 💬 GitHub Page: https://github.com/fangli333/SC-4DGS
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) has significantly elevated scene reconstruction efficiency and novel view synthesis (NVS) accuracy compared to Neural Radiance Fields (NeRF), particularly for dynamic scenes. However, current 4D NVS methods, whether based on GS or NeRF, primarily rely on camera parameters provided by COLMAP and even utilize sparse point clouds generated by COLMAP for initialization, which lack accuracy as well are time-consuming. This sometimes results in poor dynamic scene representation, especially in scenes with large object movements, or extreme camera conditions e.g. small translations combined with large rotations. Some studies simultaneously optimize the estimation of camera parameters and scenes, supervised by additional information like depth, optical flow, etc. obtained from off-the-shelf models. Using this unverified information as ground truth can reduce robustness and accuracy, which does frequently occur for long monocular videos (with e.g. > hundreds of frames). We propose a novel approach that learns a high-fidelity 4D GS scene representation with self-calibration of camera parameters. It includes the extraction of 2D point features that robustly represent 3D structure, and their use for subsequent joint optimization of camera parameters and 3D structure towards overall 4D scene optimization. We demonstrate the accuracy and time efficiency of our method through extensive quantitative and qualitative experimental results on several standard benchmarks. The results show significant improvements over state-of-the-art methods for 4D novel view synthesis. The source code will be released soon at https://github.com/fangli333/SC-4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08137v1">Survey on Fundamental Deep Learning 3D Reconstruction Techniques</a></div>
    <div class="paper-meta">
      📅 2024-07-11
    </div>
    <details class="paper-abstract">
      This survey aims to investigate fundamental deep learning (DL) based 3D reconstruction techniques that produce photo-realistic 3D models and scenes, highlighting Neural Radiance Fields (NeRFs), Latent Diffusion Models (LDM), and 3D Gaussian Splatting. We dissect the underlying algorithms, evaluate their strengths and tradeoffs, and project future research trajectories in this rapidly evolving field. We provide a comprehensive overview of the fundamental in DL-driven 3D scene reconstruction, offering insights into their potential applications and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07181v2">3D Gaussian as a New Era: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-07-10
      | 💬 Accepted at IEEE TVCG 2024, Please refer to: https://ieeexplore.ieee.org/document/10521791
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has emerged as a significant advancement in the field of Computer Graphics, offering explicit scene representation and novel view synthesis without the reliance on neural networks, such as Neural Radiance Fields (NeRF). This technique has found diverse applications in areas such as robotics, urban mapping, autonomous navigation, and virtual reality/augmented reality, just name a few. Given the growing popularity and expanding research in 3D Gaussian Splatting, this paper presents a comprehensive survey of relevant papers from the past year. We organize the survey into taxonomies based on characteristics and applications, providing an introduction to the theoretical underpinnings of 3D Gaussian Splatting. Our goal through this survey is to acquaint new researchers with 3D Gaussian Splatting, serve as a valuable reference for seminal works in the field, and inspire future research directions, as discussed in our concluding section.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07220v1">Reference-based Controllable Scene Stylization with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-09
    </div>
    <details class="paper-abstract">
      Referenced-based scene stylization that edits the appearance based on a content-aligned reference image is an emerging research area. Starting with a pretrained neural radiance field (NeRF), existing methods typically learn a novel appearance that matches the given style. Despite their effectiveness, they inherently suffer from time-consuming volume rendering, and thus are impractical for many real-time applications. In this work, we propose ReGS, which adapts 3D Gaussian Splatting (3DGS) for reference-based stylization to enable real-time stylized view synthesis. Editing the appearance of a pretrained 3DGS is challenging as it uses discrete Gaussians as 3D representation, which tightly bind appearance with geometry. Simply optimizing the appearance as prior methods do is often insufficient for modeling continuous textures in the given reference image. To address this challenge, we propose a novel texture-guided control mechanism that adaptively adjusts local responsible Gaussians to a new geometric arrangement, serving for desired texture details. The proposed process is guided by texture clues for effective appearance editing, and regularized by scene depth for preserving original geometric structure. With these novel designs, we show ReGs can produce state-of-the-art stylization results that respect the reference texture while embracing real-time rendering speed for free-view navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08551v5">GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-09
      | 💬 Accepted by ECCV 2024. Project Page:https://xingtongge.github.io/GaussianImage-page/ Code: https://github.com/Xinjie-Q/GaussianImage
    </div>
    <details class="paper-abstract">
      Implicit neural representations (INRs) recently achieved great success in image representation and compression, offering high visual quality and fast rendering speeds with 10-1000 FPS, assuming sufficient GPU resources are available. However, this requirement often hinders their use on low-end devices with limited memory. In response, we propose a groundbreaking paradigm of image representation and compression by 2D Gaussian Splatting, named GaussianImage. We first introduce 2D Gaussian to represent the image, where each Gaussian has 8 parameters including position, covariance and color. Subsequently, we unveil a novel rendering algorithm based on accumulated summation. Remarkably, our method with a minimum of 3$\times$ lower GPU memory usage and 5$\times$ faster fitting time not only rivals INRs (e.g., WIRE, I-NGP) in representation performance, but also delivers a faster rendering speed of 1500-2000 FPS regardless of parameter size. Furthermore, we integrate existing vector quantization technique to build an image codec. Experimental results demonstrate that our codec attains rate-distortion performance comparable to compression-based INRs such as COIN and COIN++, while facilitating decoding speeds of approximately 2000 FPS. Additionally, preliminary proof of concept shows that our codec surpasses COIN and COIN++ in performance when using partial bits-back coding. Code is available at https://github.com/Xinjie-Q/GaussianImage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00732v2">Gaussian Grouping: Segment and Edit Anything in 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2024-07-08
      | 💬 ECCV 2024. Gaussian Grouping extends Gaussian Splatting to fine-grained open-world 3D scene understanding. Github: https://github.com/lkeab/gaussian-grouping
    </div>
    <details class="paper-abstract">
      The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by Segment Anything Model (SAM), along with introduced 3D spatial consistency regularization. Compared to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization, style transfer and scene recomposition. Our code and models are at https://github.com/lkeab/gaussian-grouping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05324v1">PICA: Physics-Integrated Clothed Avatar</a></div>
    <div class="paper-meta">
      📅 2024-07-07
      | 💬 Project page: https://ustc3dv.github.io/PICA/
    </div>
    <details class="paper-abstract">
      We introduce PICA, a novel representation for high-fidelity animatable clothed human avatars with physics-accurate dynamics, even for loose clothing. Previous neural rendering-based representations of animatable clothed humans typically employ a single model to represent both the clothing and the underlying body. While efficient, these approaches often fail to accurately represent complex garment dynamics, leading to incorrect deformations and noticeable rendering artifacts, especially for sliding or loose garments. Furthermore, previous works represent garment dynamics as pose-dependent deformations and facilitate novel pose animations in a data-driven manner. This often results in outcomes that do not faithfully represent the mechanics of motion and are prone to generating artifacts in out-of-distribution poses. To address these issues, we adopt two individual 3D Gaussian Splatting (3DGS) models with different deformation characteristics, modeling the human body and clothing separately. This distinction allows for better handling of their respective motion characteristics. With this representation, we integrate a graph neural network (GNN)-based clothed body physics simulation module to ensure an accurate representation of clothing dynamics. Our method, through its carefully designed features, achieves high-fidelity rendering of clothed human bodies in complex and novel driving poses, significantly outperforming previous methods under the same settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05254v1">GaussReg: Fast 3D Registration with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-07
      | 💬 ECCV 2024
    </div>
    <details class="paper-abstract">
      Point cloud registration is a fundamental problem for large-scale 3D scene scanning and reconstruction. With the help of deep learning, registration methods have evolved significantly, reaching a nearly-mature stage. As the introduction of Neural Radiance Fields (NeRF), it has become the most popular 3D scene representation as its powerful view synthesis capabilities. Regarding NeRF representation, its registration is also required for large-scale scene reconstruction. However, this topic extremly lacks exploration. This is due to the inherent challenge to model the geometric relationship among two scenes with implicit representations. The existing methods usually convert the implicit representation to explicit representation for further registration. Most recently, Gaussian Splatting (GS) is introduced, employing explicit 3D Gaussian. This method significantly enhances rendering speed while maintaining high rendering quality. Given two scenes with explicit GS representations, in this work, we explore the 3D registration task between them. To this end, we propose GaussReg, a novel coarse-to-fine framework, both fast and accurate. The coarse stage follows existing point cloud registration methods and estimates a rough alignment for point clouds from GS. We further newly present an image-guided fine registration approach, which renders images from GS to provide more detailed geometric information for precise alignment. To support comprehensive evaluation, we carefully build a scene-level dataset called ScanNet-GSReg with 1379 scenes obtained from the ScanNet dataset and collect an in-the-wild dataset called GSReg. Experimental results demonstrate our method achieves state-of-the-art performance on multiple datasets. Our GaussReg is 44 times faster than HLoc (SuperPoint as the feature extractor and SuperGlue as the matcher) with comparable accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05023v1">SurgicalGaussian: Deformable 3D Gaussians for High-Fidelity Surgical Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-07-06
    </div>
    <details class="paper-abstract">
      Dynamic reconstruction of deformable tissues in endoscopic video is a key technology for robot-assisted surgery. Recent reconstruction methods based on neural radiance fields (NeRFs) have achieved remarkable results in the reconstruction of surgical scenes. However, based on implicit representation, NeRFs struggle to capture the intricate details of objects in the scene and cannot achieve real-time rendering. In addition, restricted single view perception and occluded instruments also propose special challenges in surgical scene reconstruction. To address these issues, we develop SurgicalGaussian, a deformable 3D Gaussian Splatting method to model dynamic surgical scenes. Our approach models the spatio-temporal features of soft tissues at each time stamp via a forward-mapping deformation MLP and regularization to constrain local 3D Gaussians to comply with consistent movement. With the depth initialization strategy and tool mask-guided training, our method can remove surgical instruments and reconstruct high-fidelity surgical scenes. Through experiments on various surgical videos, our network outperforms existing method on many aspects, including rendering quality, rendering speed and GPU usage. The project page can be found at https://surgicalgaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04687v2">Z-Splat: Z-Axis Gaussian Splatting for Camera-Sonar Fusion</a></div>
    <div class="paper-meta">
      📅 2024-07-05
    </div>
    <details class="paper-abstract">
      Differentiable 3D-Gaussian splatting (GS) is emerging as a prominent technique in computer vision and graphics for reconstructing 3D scenes. GS represents a scene as a set of 3D Gaussians with varying opacities and employs a computationally efficient splatting operation along with analytical derivatives to compute the 3D Gaussian parameters given scene images captured from various viewpoints. Unfortunately, capturing surround view ($360^{\circ}$ viewpoint) images is impossible or impractical in many real-world imaging scenarios, including underwater imaging, rooms inside a building, and autonomous navigation. In these restricted baseline imaging scenarios, the GS algorithm suffers from a well-known 'missing cone' problem, which results in poor reconstruction along the depth axis. In this manuscript, we demonstrate that using transient data (from sonars) allows us to address the missing cone problem by sampling high-frequency data along the depth axis. We extend the Gaussian splatting algorithms for two commonly used sonars and propose fusion algorithms that simultaneously utilize RGB camera data and sonar data. Through simulations, emulations, and hardware experiments across various imaging scenarios, we show that the proposed fusion algorithms lead to significantly better novel view synthesis (5 dB improvement in PSNR) and 3D geometry reconstruction (60% lower Chamfer distance).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15264v2">TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-05
      | 💬 Accepted at ECCV 2024. Project page: https://fictionarry.github.io/TalkingGaussian/
    </div>
    <details class="paper-abstract">
      Radiance fields have demonstrated impressive performance in synthesizing lifelike 3D talking heads. However, due to the difficulty in fitting steep appearance changes, the prevailing paradigm that presents facial motions by directly modifying point appearance may lead to distortions in dynamic regions. To tackle this challenge, we introduce TalkingGaussian, a deformation-based radiance fields framework for high-fidelity talking head synthesis. Leveraging the point-based Gaussian Splatting, facial motions can be represented in our method by applying smooth and continuous deformations to persistent Gaussian primitives, without requiring to learn the difficult appearance change like previous methods. Due to this simplification, precise facial motions can be synthesized while keeping a highly intact facial feature. Under such a deformation paradigm, we further identify a face-mouth motion inconsistency that would affect the learning of detailed speaking motions. To address this conflict, we decompose the model into two branches separately for the face and inside mouth areas, therefore simplifying the learning tasks to help reconstruct more accurate motion and structure of the mouth region. Extensive experiments demonstrate that our method renders high-quality lip-synchronized talking head videos, with better facial fidelity and higher efficiency compared with previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.12897v2">A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-07-04
      | 💬 17 pages, 11 figures, ECCV 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown remarkable success in synthesizing novel views given multiple views of a static scene. Yet, 3DGS faces challenges when applied to dynamic scenes because 3D Gaussian parameters need to be updated per timestep, requiring a large amount of memory and at least a dozen observations per timestep. To address these limitations, we present a compact dynamic 3D Gaussian representation that models positions and rotations as functions of time with a few parameter approximations while keeping other properties of 3DGS including scale, color and opacity invariant. Our method can dramatically reduce memory usage and relax a strict multi-view assumption. In our experiments on monocular and multi-view scenarios, we show that our method not only matches state-of-the-art methods, often linked with slower rendering speeds, in terms of high rendering quality but also significantly surpasses them by achieving a rendering speed of $118$ frames per second (FPS) at a resolution of 1,352$\times$1,014 on a single GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03857v1">PFGS: High Fidelity Point Cloud Rendering via Feature Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      Rendering high-fidelity images from sparse point clouds is still challenging. Existing learning-based approaches suffer from either hole artifacts, missing details, or expensive computations. In this paper, we propose a novel framework to render high-quality images from sparse points. This method first attempts to bridge the 3D Gaussian Splatting and point cloud rendering, which includes several cascaded modules. We first use a regressor to estimate Gaussian properties in a point-wise manner, the estimated properties are used to rasterize neural feature descriptors into 2D planes which are extracted from a multiscale extractor. The projected feature volume is gradually decoded toward the final prediction via a multiscale and progressive decoder. The whole pipeline experiences a two-stage training and is driven by our well-designed progressive and multiscale reconstruction loss. Experiments on different benchmarks show the superiority of our method in terms of rendering qualities and the necessities of our main components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02598v2">AutoSplat: Constrained Gaussian Splatting for Autonomous Driving Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-07-04
    </div>
    <details class="paper-abstract">
      Realistic scene reconstruction and view synthesis are essential for advancing autonomous driving systems by simulating safety-critical scenarios. 3D Gaussian Splatting excels in real-time rendering and static scene reconstructions but struggles with modeling driving scenarios due to complex backgrounds, dynamic objects, and sparse views. We propose AutoSplat, a framework employing Gaussian splatting to achieve highly realistic reconstructions of autonomous driving scenes. By imposing geometric constraints on Gaussians representing the road and sky regions, our method enables multi-view consistent simulation of challenging scenarios including lane changes. Leveraging 3D templates, we introduce a reflected Gaussian consistency constraint to supervise both the visible and unseen side of foreground objects. Moreover, to model the dynamic appearance of foreground objects, we estimate residual spherical harmonics for each foreground Gaussian. Extensive experiments on Pandaset and KITTI demonstrate that AutoSplat outperforms state-of-the-art methods in scene reconstruction and novel view synthesis across diverse driving scenarios. Visit our project page at https://autosplat.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02918v1">Free-SurGS: SfM-Free 3D Gaussian Splatting for Surgical Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-07-03
      | 💬 Accepted to MICCAI 2024
    </div>
    <details class="paper-abstract">
      Real-time 3D reconstruction of surgical scenes plays a vital role in computer-assisted surgery, holding a promise to enhance surgeons' visibility. Recent advancements in 3D Gaussian Splatting (3DGS) have shown great potential for real-time novel view synthesis of general scenes, which relies on accurate poses and point clouds generated by Structure-from-Motion (SfM) for initialization. However, 3DGS with SfM fails to recover accurate camera poses and geometry in surgical scenes due to the challenges of minimal textures and photometric inconsistencies. To tackle this problem, in this paper, we propose the first SfM-free 3DGS-based method for surgical scene reconstruction by jointly optimizing the camera poses and scene representation. Based on the video continuity, the key of our method is to exploit the immediate optical flow priors to guide the projection flow derived from 3D Gaussians. Unlike most previous methods relying on photometric loss only, we formulate the pose estimation problem as minimizing the flow loss between the projection flow and optical flow. A consistency check is further introduced to filter the flow outliers by detecting the rigid and reliable points that satisfy the epipolar geometry. During 3D Gaussian optimization, we randomly sample frames to optimize the scene representations to grow the 3D Gaussian progressively. Experiments on the SCARED dataset demonstrate our superior performance over existing methods in novel view synthesis and pose estimation with high efficiency. Code is available at https://github.com/wrld/Free-SurGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03307v3">4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2024-07-02
      | 💬 Proc. SIGGRAPH, 2024
    </div>
    <details class="paper-abstract">
      We consider the problem of novel-view synthesis (NVS) for dynamic scenes. Recent neural approaches have accomplished exceptional NVS results for static 3D scenes, but extensions to 4D time-varying scenes remain non-trivial. Prior efforts often encode dynamics by learning a canonical space plus implicit or explicit deformation fields, which struggle in challenging scenarios like sudden movements or generating high-fidelity renderings. In this paper, we introduce 4D Gaussian Splatting (4DRotorGS), a novel method that represents dynamic scenes with anisotropic 4D XYZT Gaussians, inspired by the success of 3D Gaussian Splatting in static scenes. We model dynamics at each timestamp by temporally slicing the 4D Gaussians, which naturally compose dynamic 3D Gaussians and can be seamlessly projected into images. As an explicit spatial-temporal representation, 4DRotorGS demonstrates powerful capabilities for modeling complicated dynamics and fine details--especially for scenes with abrupt motions. We further implement our temporal slicing and splatting techniques in a highly optimized CUDA acceleration framework, achieving real-time inference rendering speeds of up to 277 FPS on an RTX 3090 GPU and 583 FPS on an RTX 4090 GPU. Rigorous evaluations on scenes with diverse motions showcase the superior efficiency and effectiveness of 4DRotorGS, which consistently outperforms existing methods both quantitatively and qualitatively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01761v1">DRAGON: Drone and Ground Gaussian Splatting for 3D Building Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 12 pages, 9 figures, accepted to ICCP 2024
    </div>
    <details class="paper-abstract">
      3D building reconstruction from imaging data is an important task for many applications ranging from urban planning to reconnaissance. Modern Novel View synthesis (NVS) methods like NeRF and Gaussian Splatting offer powerful techniques for developing 3D models from natural 2D imagery in an unsupervised fashion. These algorithms generally require input training views surrounding the scene of interest, which, in the case of large buildings, is typically not available across all camera elevations. In particular, the most readily available camera viewpoints at scale across most buildings are at near-ground (e.g., with mobile phones) and aerial (drones) elevations. However, due to the significant difference in viewpoint between drone and ground image sets, camera registration - a necessary step for NVS algorithms - fails. In this work we propose a method, DRAGON, that can take drone and ground building imagery as input and produce a 3D NVS model. The key insight of DRAGON is that intermediate elevation imagery may be extrapolated by an NVS algorithm itself in an iterative procedure with perceptual regularization, thereby bridging the visual feature gap between the two elevations and enabling registration. We compiled a semi-synthetic dataset of 9 large building scenes using Google Earth Studio, and quantitatively and qualitatively demonstrate that DRAGON can generate compelling renderings on this dataset compared to baseline strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01301v1">GaussianStego: A Generalizable Stenography Pipeline for Generative 3D Gaussians Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 Project website: https://gaussian-stego.github.io/
    </div>
    <details class="paper-abstract">
      Recent advancements in large generative models and real-time neural rendering using point-based techniques pave the way for a future of widespread visual data distribution through sharing synthesized 3D assets. However, while standardized methods for embedding proprietary or copyright information, either overtly or subtly, exist for conventional visual content such as images and videos, this issue remains unexplored for emerging generative 3D formats like Gaussian Splatting. We present GaussianStego, a method for embedding steganographic information in the rendering of generated 3D assets. Our approach employs an optimization framework that enables the accurate extraction of hidden information from images rendered using Gaussian assets derived from large models, while maintaining their original visual quality. We conduct preliminary evaluations of our method across several potential deployment scenarios and discuss issues identified through analysis. GaussianStego represents an initial exploration into the novel challenge of embedding customizable, imperceptible, and recoverable information within the renders produced by current 3D generative models, while ensuring minimal impact on the rendered content's quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01029v1">EndoSparse: Real-Time Sparse View Synthesis of Endoscopic Scenes using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-07-01
      | 💬 Accpeted by MICCAI2024
    </div>
    <details class="paper-abstract">
      3D reconstruction of biological tissues from a collection of endoscopic images is a key to unlock various important downstream surgical applications with 3D capabilities. Existing methods employ various advanced neural rendering techniques for photorealistic view synthesis, but they often struggle to recover accurate 3D representations when only sparse observations are available, which is usually the case in real-world clinical scenarios. To tackle this {sparsity} challenge, we propose a framework leveraging the prior knowledge from multiple foundation models during the reconstruction process, dubbed as \textit{EndoSparse}. Experimental results indicate that our proposed strategy significantly improves the geometric and appearance quality under challenging sparse-view conditions, including using only three views. In rigorous benchmarking experiments against state-of-the-art methods, \textit{EndoSparse} achieves superior results in terms of accurate geometry, realistic appearance, and rendering efficiency, confirming the robustness to sparse-view limitations in endoscopic reconstruction. \textit{EndoSparse} signifies a steady step towards the practical deployment of neural 3D reconstruction in real-world clinical scenarios. Project page: https://endo-sparse.github.io/.
    </details>
</div>
