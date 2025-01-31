# gaussian splatting - 2024_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00362v1">UDGS-SLAM : UniDepth Assisted Gaussian Splatting for Monocular SLAM</a></div>
    <div class="paper-meta">
      📅 2024-08-31
    </div>
    <details class="paper-abstract">
      Recent advancements in monocular neural depth estimation, particularly those achieved by the UniDepth network, have prompted the investigation of integrating UniDepth within a Gaussian splatting framework for monocular SLAM.This study presents UDGS-SLAM, a novel approach that eliminates the necessity of RGB-D sensors for depth estimation within Gaussian splatting framework. UDGS-SLAM employs statistical filtering to ensure local consistency of the estimated depth and jointly optimizes camera trajectory and Gaussian scene representation parameters. The proposed method achieves high-fidelity rendered images and low ATERMSE of the camera trajectory. The performance of UDGS-SLAM is rigorously evaluated using the TUM RGB-D dataset and benchmarked against several baseline methods, demonstrating superior performance across various scenarios. Additionally, an ablation study is conducted to validate design choices and investigate the impact of different network backbone encoders on system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18033v2">RT-GS2: Real-Time Generalizable Semantic Segmentation for 3D Gaussian Representations of Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2024-08-30
      | 💬 Accepted paper at BMVC 2024
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has revolutionized the world of novel view synthesis by achieving high rendering performance in real-time. Recently, studies have focused on enriching these 3D representations with semantic information for downstream tasks. In this paper, we introduce RT-GS2, the first generalizable semantic segmentation method employing Gaussian Splatting. While existing Gaussian Splatting-based approaches rely on scene-specific training, RT-GS2 demonstrates the ability to generalize to unseen scenes. Our method adopts a new approach by first extracting view-independent 3D Gaussian features in a self-supervised manner, followed by a novel View-Dependent / View-Independent (VDVI) feature fusion to enhance semantic consistency over different views. Extensive experimentation on three different datasets showcases RT-GS2's superiority over the state-of-the-art methods in semantic segmentation quality, exemplified by a 8.01% increase in mIoU on the Replica dataset. Moreover, our method achieves real-time performance of 27.03 FPS, marking an astonishing 901 times speedup compared to existing approaches. This work represents a significant advancement in the field by introducing, to the best of our knowledge, the first real-time generalizable semantic segmentation method for 3D Gaussian representations of radiance fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00583v2">DeformGS: Scene Flow in Highly Deformable Scenes for Deformable Object Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-08-30
    </div>
    <details class="paper-abstract">
      Teaching robots to fold, drape, or reposition deformable objects such as cloth will unlock a variety of automation applications. While remarkable progress has been made for rigid object manipulation, manipulating deformable objects poses unique challenges, including frequent occlusions, infinite-dimensional state spaces and complex dynamics. Just as object pose estimation and tracking have aided robots for rigid manipulation, dense 3D tracking (scene flow) of highly deformable objects will enable new applications in robotics while aiding existing approaches, such as imitation learning or creating digital twins with real2sim transfer. We propose DeformGS, an approach to recover scene flow in highly deformable scenes, using simultaneous video captures of a dynamic scene from multiple cameras. DeformGS builds on recent advances in Gaussian splatting, a method that learns the properties of a large number of Gaussians for state-of-the-art and fast novel-view synthesis. DeformGS learns a deformation function to project a set of Gaussians with canonical properties into world space. The deformation function uses a neural-voxel encoding and a multilayer perceptron (MLP) to infer Gaussian position, rotation, and a shadow scalar. We enforce physics-inspired regularization terms based on conservation of momentum and isometry, which leads to trajectories with smaller trajectory errors. We also leverage existing foundation models SAM and XMEM to produce noisy masks, and learn a per-Gaussian mask for better physics-inspired regularization. DeformGS achieves high-quality 3D tracking on highly deformable scenes with shadows and occlusions. In experiments, DeformGS improves 3D tracking by an average of 55.8% compared to the state-of-the-art. With sufficient texture, DeformGS achieves a median tracking error of 3.3 mm on a cloth of 1.5 x 1.5 m in area. Website: https://deformgs.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17223v1">OG-Mapping: Octree-based Structured 3D Gaussians for Online Dense Mapping</a></div>
    <div class="paper-meta">
      📅 2024-08-30
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has recently demonstrated promising advancements in RGB-D online dense mapping. Nevertheless, existing methods excessively rely on per-pixel depth cues to perform map densification, which leads to significant redundancy and increased sensitivity to depth noise. Additionally, explicitly storing 3D Gaussian parameters of room-scale scene poses a significant storage challenge. In this paper, we introduce OG-Mapping, which leverages the robust scene structural representation capability of sparse octrees, combined with structured 3D Gaussian representations, to achieve efficient and robust online dense mapping. Moreover, OG-Mapping employs an anchor-based progressive map refinement strategy to recover the scene structures at multiple levels of detail. Instead of maintaining a small number of active keyframes with a fixed keyframe window as previous approaches do, a dynamic keyframe window is employed to allow OG-Mapping to better tackle false local minima and forgetting issues. Experimental results demonstrate that OG-Mapping delivers more robust and superior realism mapping results than existing Gaussian-based RGB-D online mapping methods with a compact model, and no additional post-processing is required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16982v1">2DGH: 2D Gaussian-Hermite Splatting for High-quality Rendering and Better Geometry Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-08-30
    </div>
    <details class="paper-abstract">
      2D Gaussian Splatting has recently emerged as a significant method in 3D reconstruction, enabling novel view synthesis and geometry reconstruction simultaneously. While the well-known Gaussian kernel is broadly used, its lack of anisotropy and deformation ability leads to dim and vague edges at object silhouettes, limiting the reconstruction quality of current Gaussian splatting methods. To enhance the representation power, we draw inspiration from quantum physics and propose to use the Gaussian-Hermite kernel as the new primitive in Gaussian splatting. The new kernel takes a unified mathematical form and extends the Gaussian function, which serves as the zero-rank term in the updated formulation. Our experiments demonstrate the extraordinary performance of Gaussian-Hermite kernel in both geometry reconstruction and novel-view synthesis tasks. The proposed kernel outperforms traditional Gaussian Splatting kernels, showcasing its potential for high-quality 3D reconstruction and rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16760v1">OmniRe: Omni Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-08-29
      | 💬 See the project page for code, video results and demos: https://ziyc.github.io/omnire/
    </div>
    <details class="paper-abstract">
      We introduce OmniRe, a holistic approach for efficiently reconstructing high-fidelity dynamic urban scenes from on-device logs. Recent methods for modeling driving sequences using neural radiance fields or Gaussian Splatting have demonstrated the potential of reconstructing challenging dynamic scenes, but often overlook pedestrians and other non-vehicle dynamic actors, hindering a complete pipeline for dynamic urban scene reconstruction. To that end, we propose a comprehensive 3DGS framework for driving scenes, named OmniRe, that allows for accurate, full-length reconstruction of diverse dynamic objects in a driving log. OmniRe builds dynamic neural scene graphs based on Gaussian representations and constructs multiple local canonical spaces that model various dynamic actors, including vehicles, pedestrians, and cyclists, among many others. This capability is unmatched by existing methods. OmniRe allows us to holistically reconstruct different objects present in the scene, subsequently enabling the simulation of reconstructed scenarios with all actors participating in real-time (~60Hz). Extensive evaluations on the Waymo dataset show that our approach outperforms prior state-of-the-art methods quantitatively and qualitatively by a large margin. We believe our work fills a critical gap in driving reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02255v3">Re-Nerfing: Improving Novel View Synthesis through Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-08-28
      | 💬 Code will be released upon acceptance
    </div>
    <details class="paper-abstract">
      Recent neural rendering and reconstruction techniques, such as NeRFs or Gaussian Splatting, have shown remarkable novel view synthesis capabilities but require hundreds of images of the scene from diverse viewpoints to render high-quality novel views. With fewer images available, these methods start to fail since they can no longer correctly triangulate the underlying 3D geometry and converge to a non-optimal solution. These failures can manifest as floaters or blurry renderings in sparsely observed areas of the scene. In this paper, we propose Re-Nerfing, a simple and general add-on approach that leverages novel view synthesis itself to tackle this problem. Using an already trained NVS method, we render novel views between existing ones and augment the training data to optimize a second model. This introduces additional multi-view constraints and allows the second model to converge to a better solution. With Re-Nerfing we achieve significant improvements upon multiple pipelines based on NeRF and Gaussian-Splatting in sparse view settings of the mip-NeRF 360 and LLFF datasets. Notably, Re-Nerfing does not require prior knowledge or extra supervision signals, making it a flexible and practical add-on.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15708v1">Towards Realistic Example-based Modeling via 3D Gaussian Stitching</a></div>
    <div class="paper-meta">
      📅 2024-08-28
    </div>
    <details class="paper-abstract">
      Using parts of existing models to rebuild new models, commonly termed as example-based modeling, is a classical methodology in the realm of computer graphics. Previous works mostly focus on shape composition, making them very hard to use for realistic composition of 3D objects captured from real-world scenes. This leads to combining multiple NeRFs into a single 3D scene to achieve seamless appearance blending. However, the current SeamlessNeRF method struggles to achieve interactive editing and harmonious stitching for real-world scenes due to its gradient-based strategy and grid-based representation. To this end, we present an example-based modeling method that combines multiple Gaussian fields in a point-based representation using sample-guided synthesis. Specifically, as for composition, we create a GUI to segment and transform multiple fields in real time, easily obtaining a semantically meaningful composition of models represented by 3D Gaussian Splatting (3DGS). For texture blending, due to the discrete and irregular nature of 3DGS, straightforwardly applying gradient propagation as SeamlssNeRF is not supported. Thus, a novel sampling-based cloning method is proposed to harmonize the blending while preserving the original rich texture and content. Our workflow consists of three steps: 1) real-time segmentation and transformation of a Gaussian model using a well-tailored GUI, 2) KNN analysis to identify boundary points in the intersecting area between the source and target models, and 3) two-phase optimization of the target model using sampling-based cloning and gradient constraints. Extensive experimental results validate that our approach significantly outperforms previous works in terms of realistic synthesis, demonstrating its practicality. More demos are available at https://ingra14m.github.io/gs_stitching_website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15891v4">OMEGAS: Object Mesh Extraction from Large Scenes Guided by Gaussian Segmentation</a></div>
    <div class="paper-meta">
      📅 2024-08-27
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D reconstruction technologies have paved the way for high-quality and real-time rendering of complex 3D scenes. Despite these achievements, a notable challenge persists: it is difficult to precisely reconstruct specific objects from large scenes. Current scene reconstruction techniques frequently result in the loss of object detail textures and are unable to reconstruct object portions that are occluded or unseen in views. To address this challenge, we delve into the meticulous 3D reconstruction of specific objects within large scenes and propose a framework termed OMEGAS: Object Mesh Extraction from Large Scenes Guided by Gaussian Segmentation. Specifically, we proposed a novel 3D target segmentation technique based on 2D Gaussian Splatting, which segments 3D consistent target masks in multi-view scene images and generates a preliminary target model. Moreover, to reconstruct the unseen portions of the target, we propose a novel target replenishment technique driven by large-scale generative diffusion priors. We demonstrate that our method can accurately reconstruct specific targets from large scenes, both quantitatively and qualitatively. Our experiments show that OMEGAS significantly outperforms existing reconstruction methods across various scenarios. Our project page is at: https://github.com/CrystalWlz/OMEGAS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13912v2">Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs</a></div>
    <div class="paper-meta">
      📅 2024-08-27
      | 💬 Our project page can be found at: https://splatt3r.active.vision/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce Splatt3R, a pose-free, feed-forward method for in-the-wild 3D reconstruction and novel view synthesis from stereo pairs. Given uncalibrated natural images, Splatt3R can predict 3D Gaussian Splats without requiring any camera parameters or depth information. For generalizability, we build Splatt3R upon a ``foundation'' 3D geometry reconstruction method, MASt3R, by extending it to deal with both 3D structure and appearance. Specifically, unlike the original MASt3R which reconstructs only 3D point clouds, we predict the additional Gaussian attributes required to construct a Gaussian primitive for each point. Hence, unlike other novel view synthesis methods, Splatt3R is first trained by optimizing the 3D point cloud's geometry loss, and then a novel view synthesis objective. By doing this, we avoid the local minima present in training 3D Gaussian Splats from stereo views. We also propose a novel loss masking strategy that we empirically find is critical for strong performance on extrapolated viewpoints. We train Splatt3R on the ScanNet++ dataset and demonstrate excellent generalisation to uncalibrated, in-the-wild images. Splatt3R can reconstruct scenes at 4FPS at 512 x 512 resolution, and the resultant splats can be rendered in real-time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15242v1">Drone-assisted Road Gaussian Splatting with Cross-view Uncertainty</a></div>
    <div class="paper-meta">
      📅 2024-08-27
      | 💬 BMVC2024 Project Page: https://sainingzhang.github.io/project/uc-gs/ Code: https://github.com/SainingZhang/uc-gs/
    </div>
    <details class="paper-abstract">
      Robust and realistic rendering for large-scale road scenes is essential in autonomous driving simulation. Recently, 3D Gaussian Splatting (3D-GS) has made groundbreaking progress in neural rendering, but the general fidelity of large-scale road scene renderings is often limited by the input imagery, which usually has a narrow field of view and focuses mainly on the street-level local area. Intuitively, the data from the drone's perspective can provide a complementary viewpoint for the data from the ground vehicle's perspective, enhancing the completeness of scene reconstruction and rendering. However, training naively with aerial and ground images, which exhibit large view disparity, poses a significant convergence challenge for 3D-GS, and does not demonstrate remarkable improvements in performance on road views. In order to enhance the novel view synthesis of road views and to effectively use the aerial information, we design an uncertainty-aware training method that allows aerial images to assist in the synthesis of areas where ground images have poor learning outcomes instead of weighting all pixels equally in 3D-GS training like prior work did. We are the first to introduce the cross-view uncertainty to 3D-GS by matching the car-view ensemble-based rendering uncertainty to aerial images, weighting the contribution of each pixel to the training process. Additionally, to systematically quantify evaluation metrics, we assemble a high-quality synthesized dataset comprising both aerial and ground images for road scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11413v2">Pano2Room: Novel View Synthesis from a Single Indoor Panorama</a></div>
    <div class="paper-meta">
      📅 2024-08-27
      | 💬 SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers '24), December 3--6, 2024, Tokyo, Japan
    </div>
    <details class="paper-abstract">
      Recent single-view 3D generative methods have made significant advancements by leveraging knowledge distilled from extensive 3D object datasets. However, challenges persist in the synthesis of 3D scenes from a single view, primarily due to the complexity of real-world environments and the limited availability of high-quality prior resources. In this paper, we introduce a novel approach called Pano2Room, designed to automatically reconstruct high-quality 3D indoor scenes from a single panoramic image. These panoramic images can be easily generated using a panoramic RGBD inpainter from captures at a single location with any camera. The key idea is to initially construct a preliminary mesh from the input panorama, and iteratively refine this mesh using a panoramic RGBD inpainter while collecting photo-realistic 3D-consistent pseudo novel views. Finally, the refined mesh is converted into a 3D Gaussian Splatting field and trained with the collected pseudo novel views. This pipeline enables the reconstruction of real-world 3D scenes, even in the presence of large occlusions, and facilitates the synthesis of photo-realistic novel views with detailed geometry. Extensive qualitative and quantitative experiments have been conducted to validate the superiority of our method in single-panorama indoor novel synthesis compared to the state-of-the-art. Our code and data are available at \url{https://github.com/TrickyGo/Pano2Room}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14823v1">LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive Streaming</a></div>
    <div class="paper-meta">
      📅 2024-08-27
    </div>
    <details class="paper-abstract">
      The rise of Extended Reality (XR) requires efficient streaming of 3D online worlds, challenging current 3DGS representations to adapt to bandwidth-constrained environments. This paper proposes LapisGS, a layered 3DGS that supports adaptive streaming and progressive rendering. Our method constructs a layered structure for cumulative representation, incorporates dynamic opacity optimization to maintain visual fidelity, and utilizes occupancy maps to efficiently manage Gaussian splats. This proposed model offers a progressive representation supporting a continuous rendering quality adapted for bandwidth-aware streaming. Extensive experiments validate the effectiveness of our approach in balancing visual fidelity with the compactness of the model, with up to 50.71% improvement in SSIM, 286.53% improvement in LPIPS, and 318.41% reduction in model size, and shows its potential for bandwidth-adapted 3D streaming and rendering applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00752v4">On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy</a></div>
    <div class="paper-meta">
      📅 2024-08-26
      | 💬 Accepted by ECCV2024; Project Page: https://letianhuang.github.io/op43dgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has garnered extensive attention and application in real-time neural rendering. Concurrently, concerns have been raised about the limitations of this technology in aspects such as point cloud storage, performance, and robustness in sparse viewpoints, leading to various improvements. However, there has been a notable lack of attention to the fundamental problem of projection errors introduced by the local affine approximation inherent in the splatting itself, and the consequential impact of these errors on the quality of photo-realistic rendering. This paper addresses the projection error function of 3D Gaussian Splatting, commencing with the residual error from the first-order Taylor expansion of the projection function. The analysis establishes a correlation between the error and the Gaussian mean position. Subsequently, leveraging function optimization theory, this paper analyzes the function's minima to provide an optimal projection strategy for Gaussian Splatting referred to Optimal Gaussian Splatting, which can accommodate a variety of camera models. Experimental validation further confirms that this projection methodology reduces artifacts, resulting in a more convincingly realistic rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04249v2">InstantStyleGaussian: Efficient Art Style Transfer with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-26
    </div>
    <details class="paper-abstract">
      We present InstantStyleGaussian, an innovative 3D style transfer method based on the 3D Gaussian Splatting (3DGS) scene representation. By inputting a target-style image, it quickly generates new 3D GS scenes. Our method operates on pre-reconstructed GS scenes, combining diffusion models with an improved iterative dataset update strategy. It utilizes diffusion models to generate target style images, adds these new images to the training dataset, and uses this dataset to iteratively update and optimize the GS scenes, significantly accelerating the style editing process while ensuring the quality of the generated scenes. Extensive experimental results demonstrate that our method ensures high-quality stylized scenes while offering significant advantages in style transfer speed and consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13972v1">DynaSurfGS: Dynamic Surface Reconstruction with Planar-based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-26
      | 💬 homepage: https://open3dvlab.github.io/DynaSurfGS/, code: https://github.com/Open3DVLab/DynaSurfGS
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction has garnered significant attention in recent years due to its capabilities in high-quality and real-time rendering. Among various methodologies, constructing a 4D spatial-temporal representation, such as 4D-GS, has gained popularity for its high-quality rendered images. However, these methods often produce suboptimal surfaces, as the discrete 3D Gaussian point clouds fail to align with the object's surface precisely. To address this problem, we propose DynaSurfGS to achieve both photorealistic rendering and high-fidelity surface reconstruction of dynamic scenarios. Specifically, the DynaSurfGS framework first incorporates Gaussian features from 4D neural voxels with the planar-based Gaussian Splatting to facilitate precise surface reconstruction. It leverages normal regularization to enforce the smoothness of the surface of dynamic objects. It also incorporates the as-rigid-as-possible (ARAP) constraint to maintain the approximate rigidity of local neighborhoods of 3D Gaussians between timesteps and ensure that adjacent 3D Gaussians remain closely aligned throughout. Extensive experiments demonstrate that DynaSurfGS surpasses state-of-the-art methods in both high-fidelity surface reconstruction and photorealistic rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13770v1">TranSplat: Generalizable 3D Gaussian Splatting from Sparse Multi-View Images with Transformers</a></div>
    <div class="paper-meta">
      📅 2024-08-25
    </div>
    <details class="paper-abstract">
      Compared with previous 3D reconstruction methods like Nerf, recent Generalizable 3D Gaussian Splatting (G-3DGS) methods demonstrate impressive efficiency even in the sparse-view setting. However, the promising reconstruction performance of existing G-3DGS methods relies heavily on accurate multi-view feature matching, which is quite challenging. Especially for the scenes that have many non-overlapping areas between various views and contain numerous similar regions, the matching performance of existing methods is poor and the reconstruction precision is limited. To address this problem, we develop a strategy that utilizes a predicted depth confidence map to guide accurate local feature matching. In addition, we propose to utilize the knowledge of existing monocular depth estimation models as prior to boost the depth estimation precision in non-overlapping areas between views. Combining the proposed strategies, we present a novel G-3DGS method named TranSplat, which obtains the best performance on both the RealEstate10K and ACID benchmarks while maintaining competitive speed and presenting strong cross-dataset generalization ability. Our code, and demos will be available at: https://xingyoujun.github.io/transplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13370v1">BiGS: Bidirectional Gaussian Primitives for Relightable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-23
    </div>
    <details class="paper-abstract">
      We present Bidirectional Gaussian Primitives, an image-based novel view synthesis technique designed to represent and render 3D objects with surface and volumetric materials under dynamic illumination. Our approach integrates light intrinsic decomposition into the Gaussian splatting framework, enabling real-time relighting of 3D objects. To unify surface and volumetric material within a cohesive appearance model, we adopt a light- and view-dependent scattering representation via bidirectional spherical harmonics. Our model does not use a specific surface normal-related reflectance function, making it more compatible with volumetric representations like Gaussian splatting, where the normals are undefined. We demonstrate our method by reconstructing and rendering objects with complex materials. Using One-Light-At-a-Time (OLAT) data as input, we can reproduce photorealistic appearances under novel lighting conditions in real time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05868v1">SpecGaussian with Latent Features: A High-quality Modeling of the View-dependent Appearance for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-23
      | 💬 9 pages,6 figures, 5 tables, ACM Multimedia 2024
    </div>
    <details class="paper-abstract">
      Recently, the 3D Gaussian Splatting (3D-GS) method has achieved great success in novel view synthesis, providing real-time rendering while ensuring high-quality rendering results. However, this method faces challenges in modeling specular reflections and handling anisotropic appearance components, especially in dealing with view-dependent color under complex lighting conditions. Additionally, 3D-GS uses spherical harmonic to learn the color representation, which has limited ability to represent complex scenes. To overcome these challenges, we introduce Lantent-SpecGS, an approach that utilizes a universal latent neural descriptor within each 3D Gaussian. This enables a more effective representation of 3D feature fields, including appearance and geometry. Moreover, two parallel CNNs are designed to decoder the splatting feature maps into diffuse color and specular color separately. A mask that depends on the viewpoint is learned to merge these two colors, resulting in the final rendered image. Experimental results demonstrate that our method obtains competitive performance in novel view synthesis and extends the ability of 3D-GS to handle intricate scenarios with specular reflections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12894v1">FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering</a></div>
    <div class="paper-meta">
      📅 2024-08-23
      | 💬 Project page: https://3dgs-flod.github.io/flod.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves fast and high-quality renderings by using numerous small Gaussians, which leads to significant memory consumption. This reliance on a large number of Gaussians restricts the application of 3DGS-based models on low-cost devices due to memory limitations. However, simply reducing the number of Gaussians to accommodate devices with less memory capacity leads to inferior quality compared to the quality that can be achieved on high-end hardware. To address this lack of scalability, we propose integrating a Flexible Level of Detail (FLoD) to 3DGS, to allow a scene to be rendered at varying levels of detail according to hardware capabilities. While existing 3DGSs with LoD focus on detailed reconstruction, our method provides reconstructions using a small number of Gaussians for reduced memory requirements, and a larger number of Gaussians for greater detail. Experiments demonstrate our various rendering options with tradeoffs between rendering quality and memory usage, thereby allowing real-time rendering across different memory constraints. Furthermore, we show that our method generalizes to different 3DGS frameworks, indicating its potential for integration into future state-of-the-art developments. Project page: https://3dgs-flod.github.io/flod.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15624v2">Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-23
      | 💬 Project page: see https://semantic-gaussians.github.io
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D scene understanding presents a significant challenge in computer vision, with wide-ranging applications in embodied agents and augmented reality systems. Existing methods adopt neurel rendering methods as 3D representations and jointly optimize color and semantic features to achieve rendering and scene understanding simultaneously. In this paper, we introduce Semantic Gaussians, a novel open-vocabulary scene understanding approach based on 3D Gaussian Splatting. Our key idea is to distill knowledge from 2D pre-trained models to 3D Gaussians. Unlike existing methods, we design a versatile projection approach that maps various 2D semantic features from pre-trained image encoders into a novel semantic component of 3D Gaussians, which is based on spatial relationship and need no additional training. We further build a 3D semantic network that directly predicts the semantic component from raw 3D Gaussians for fast inference. The quantitative results on ScanNet segmentation and LERF object localization demonstates the superior performance of our method. Additionally, we explore several applications of Semantic Gaussians including object part segmentation, instance segmentation, scene editing, and spatiotemporal segmentation with better qualitative results over 2D and 3D baselines, highlighting its versatility and effectiveness on supporting diverse downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11697v1">Robust 3D Gaussian Splatting for Novel View Synthesis in Presence of Distractors</a></div>
    <div class="paper-meta">
      📅 2024-08-21
      | 💬 GCPR 2024, Project Page: https://paulungermann.github.io/Robust3DGaussians , Video: https://www.youtube.com/watch?v=P9unyR7yK3E
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has shown impressive novel view synthesis results; nonetheless, it is vulnerable to dynamic objects polluting the input data of an otherwise static scene, so called distractors. Distractors have severe impact on the rendering quality as they get represented as view-dependent effects or result in floating artifacts. Our goal is to identify and ignore such distractors during the 3D Gaussian optimization to obtain a clean reconstruction. To this end, we take a self-supervised approach that looks at the image residuals during the optimization to determine areas that have likely been falsified by a distractor. In addition, we leverage a pretrained segmentation network to provide object awareness, enabling more accurate exclusion of distractors. This way, we obtain segmentation masks of distractors to effectively ignore them in the loss formulation. We demonstrate that our approach is robust to various distractors and strongly improves rendering quality on distractor-polluted scenes, improving PSNR by 1.86dB compared to 3D Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05635v2">Visual SLAM with 3D Gaussian Primitives and Depth Priors Enabling Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-08-21
    </div>
    <details class="paper-abstract">
      Conventional geometry-based SLAM systems lack dense 3D reconstruction capabilities since their data association usually relies on feature correspondences. Additionally, learning-based SLAM systems often fall short in terms of real-time performance and accuracy. Balancing real-time performance with dense 3D reconstruction capabilities is a challenging problem. In this paper, we propose a real-time RGB-D SLAM system that incorporates a novel view synthesis technique, 3D Gaussian Splatting, for 3D scene representation and pose estimation. This technique leverages the real-time rendering performance of 3D Gaussian Splatting with rasterization and allows for differentiable optimization in real time through CUDA implementation. We also enable mesh reconstruction from 3D Gaussians for explicit dense 3D reconstruction. To estimate accurate camera poses, we utilize a rotation-translation decoupled strategy with inverse optimization. This involves iteratively updating both in several iterations through gradient-based optimization. This process includes differentiably rendering RGB, depth, and silhouette maps and updating the camera parameters to minimize a combined loss of photometric loss, depth geometry loss, and visibility loss, given the existing 3D Gaussian map. However, 3D Gaussian Splatting (3DGS) struggles to accurately represent surfaces due to the multi-view inconsistency of 3D Gaussians, which can lead to reduced accuracy in both camera pose estimation and scene reconstruction. To address this, we utilize depth priors as additional regularization to enforce geometric constraints, thereby improving the accuracy of both pose estimation and 3D reconstruction. We also provide extensive experimental results on public benchmark datasets to demonstrate the effectiveness of our proposed methods in terms of pose accuracy, geometric accuracy, and rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02034v2">TrAME: Trajectory-Anchored Multi-View Editing for Text-Guided 3D Gaussian Splatting Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-08-21
    </div>
    <details class="paper-abstract">
      Despite significant strides in the field of 3D scene editing, current methods encounter substantial challenge, particularly in preserving 3D consistency in multi-view editing process. To tackle this challenge, we propose a progressive 3D editing strategy that ensures multi-view consistency via a Trajectory-Anchored Scheme (TAS) with a dual-branch editing mechanism. Specifically, TAS facilitates a tightly coupled iterative process between 2D view editing and 3D updating, preventing error accumulation yielded from text-to-image process. Additionally, we explore the relationship between optimization-based methods and reconstruction-based methods, offering a unified perspective for selecting superior design choice, supporting the rationale behind the designed TAS. We further present a tuning-free View-Consistent Attention Control (VCAC) module that leverages cross-view semantic and geometric reference from the source branch to yield aligned views from the target branch during the editing of 2D views. To validate the effectiveness of our method, we analyze 2D examples to demonstrate the improved consistency with the VCAC module. Further extensive quantitative and qualitative results in text-guided 3D scene editing indicate that our method achieves superior editing quality compared to state-of-the-art methods. We will make the complete codebase publicly available following the conclusion of the review process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10906v1">ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become the de facto method of 3D representation in many vision tasks. This calls for the 3D understanding directly in this representation space. To facilitate the research in this direction, we first build a large-scale dataset of 3DGS using the commonly used ShapeNet and ModelNet datasets. Our dataset ShapeSplat consists of 65K objects from 87 unique categories, whose labels are in accordance with the respective datasets. The creation of this dataset utilized the compute equivalent of 2 GPU years on a TITAN XP GPU. We utilize our dataset for unsupervised pretraining and supervised finetuning for classification and segmentation tasks. To this end, we introduce \textbf{\textit{Gaussian-MAE}}, which highlights the unique benefits of representation learning from Gaussian parameters. Through exhaustive experiments, we provide several valuable insights. In particular, we show that (1) the distribution of the optimized GS centroids significantly differs from the uniformly sampled point cloud (used for initialization) counterpart; (2) this change in distribution results in degradation in classification but improvement in segmentation tasks when using only the centroids; (3) to leverage additional Gaussian parameters, we propose Gaussian feature grouping in a normalized feature space, along with splats pooling layer, offering a tailored solution to effectively group and embed similar Gaussians, which leads to notable improvement in finetuning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09130v2">Gaussian in the Dark: Real-Time View Synthesis From Inconsistent Dark Images Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 accepted by PG 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently emerged as a powerful representation that can synthesize remarkable novel views using consistent multi-view images as input. However, we notice that images captured in dark environments where the scenes are not fully illuminated can exhibit considerable brightness variations and multi-view inconsistency, which poses great challenges to 3D Gaussian Splatting and severely degrades its performance. To tackle this problem, we propose Gaussian-DK. Observing that inconsistencies are mainly caused by camera imaging, we represent a consistent radiance field of the physical world using a set of anisotropic 3D Gaussians, and design a camera response module to compensate for multi-view inconsistencies. We also introduce a step-based gradient scaling strategy to constrain Gaussians near the camera, which turn out to be floaters, from splitting and cloning. Experiments on our proposed benchmark dataset demonstrate that Gaussian-DK produces high-quality renderings without ghosting and floater artifacts and significantly outperforms existing methods. Furthermore, we can also synthesize light-up images by controlling exposure levels that clearly show details in shadow areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10588v1">DEGAS: Detailed Expressions on Full-Body Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      Although neural rendering has made significant advancements in creating lifelike, animatable full-body and head avatars, incorporating detailed expressions into full-body avatars remains largely unexplored. We present DEGAS, the first 3D Gaussian Splatting (3DGS)-based modeling method for full-body avatars with rich facial expressions. Trained on multiview videos of a given subject, our method learns a conditional variational autoencoder that takes both the body motion and facial expression as driving signals to generate Gaussian maps in the UV layout. To drive the facial expressions, instead of the commonly used 3D Morphable Models (3DMMs) in 3D head avatars, we propose to adopt the expression latent space trained solely on 2D portrait images, bridging the gap between 2D talking faces and 3D avatars. Leveraging the rendering capability of 3DGS and the rich expressiveness of the expression latent space, the learned avatars can be reenacted to reproduce photorealistic rendering images with subtle and accurate facial expressions. Experiments on an existing dataset and our newly proposed dataset of full-body talking avatars demonstrate the efficacy of our method. We also propose an audio-driven extension of our method with the help of 2D talking faces, opening new possibilities to interactive AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10154v2">LoopSplat: Loop Closure by Registering 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 Project page: https://loopsplat.github.io/
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) based on 3D Gaussian Splats (3DGS) has recently shown promise towards more accurate, dense 3D scene maps. However, existing 3DGS-based methods fail to address the global consistency of the scene via loop closure and/or global bundle adjustment. To this end, we propose LoopSplat, which takes RGB-D images as input and performs dense mapping with 3DGS submaps and frame-to-model tracking. LoopSplat triggers loop closure online and computes relative loop edge constraints between submaps directly via 3DGS registration, leading to improvements in efficiency and accuracy over traditional global-to-local point cloud registration. It uses a robust pose graph optimization formulation and rigidly aligns the submaps to achieve global consistency. Evaluation on the synthetic Replica and real-world TUM-RGBD, ScanNet, and ScanNet++ datasets demonstrates competitive or superior tracking, mapping, and rendering compared to existing methods for dense RGB-D SLAM. Code is available at loopsplat.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00827v3">GaussianStyle: Gaussian Head Avatar via StyleGAN</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 demo page and code to be updated soon
    </div>
    <details class="paper-abstract">
      Existing methods like Neural Radiation Fields (NeRF) and 3D Gaussian Splatting (3DGS) have made significant strides in facial attribute control such as facial animation and components editing, yet they struggle with fine-grained representation and scalability in dynamic head modeling. To address these limitations, we propose GaussianStyle, a novel framework that integrates the volumetric strengths of 3DGS with the powerful implicit representation of StyleGAN. The GaussianStyle preserves structural information, such as expressions and poses, using Gaussian points, while projecting the implicit volumetric representation into StyleGAN to capture high-frequency details and mitigate the over-smoothing commonly observed in neural texture rendering. Experimental outcomes indicate that our method achieves state-of-the-art performance in reenactment, novel view synthesis, and animation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07967v2">FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering</a></div>
    <div class="paper-meta">
      📅 2024-08-19
    </div>
    <details class="paper-abstract">
      This work introduces FlashGS, an open-source CUDA Python library, designed to facilitate the efficient differentiable rasterization of 3D Gaussian Splatting through algorithmic and kernel-level optimizations. FlashGS is developed based on the observations from a comprehensive analysis of the rendering process to enhance computational efficiency and bring the technique to wide adoption. The paper includes a suite of optimization strategies, encompassing redundancy elimination, efficient pipelining, refined control and scheduling mechanisms, and memory access optimizations, all of which are meticulously integrated to amplify the performance of the rasterization process. An extensive evaluation of FlashGS' performance has been conducted across a diverse spectrum of synthetic and real-world large-scale scenes, encompassing a variety of image resolutions. The empirical findings demonstrate that FlashGS consistently achieves an average 4x acceleration over mobile consumer GPUs, coupled with reduced memory consumption. These results underscore the superior performance and resource optimization capabilities of FlashGS, positioning it as a formidable tool in the domain of 3D rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.01339v3">Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-18
      | 💬 Project page: https://zju3dv.github.io/street_gaussians/
    </div>
    <details class="paper-abstract">
      This paper aims to tackle the problem of modeling dynamic urban streets for autonomous driving scenes. Recent methods extend NeRF by incorporating tracked vehicle poses to animate vehicles, enabling photo-realistic view synthesis of dynamic urban street scenes. However, significant limitations are their slow training and rendering speed. We introduce Street Gaussians, a new explicit scene representation that tackles these limitations. Specifically, the dynamic urban scene is represented as a set of point clouds equipped with semantic logits and 3D Gaussians, each associated with either a foreground vehicle or the background. To model the dynamics of foreground object vehicles, each object point cloud is optimized with optimizable tracked poses, along with a 4D spherical harmonics model for the dynamic appearance. The explicit representation allows easy composition of object vehicles and background, which in turn allows for scene editing operations and rendering at 135 FPS (1066 $\times$ 1600 resolution) within half an hour of training. The proposed method is evaluated on multiple challenging benchmarks, including KITTI and Waymo Open datasets. Experiments show that the proposed method consistently outperforms state-of-the-art methods across all datasets. The code will be released to ensure reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16600v3">DHGS: Decoupled Hybrid Gaussian Splatting for Driving Scene</a></div>
    <div class="paper-meta">
      📅 2024-08-18
      | 💬 13 pages, 14 figures, conference
    </div>
    <details class="paper-abstract">
      Existing Gaussian splatting methods often fall short in achieving satisfactory novel view synthesis in driving scenes, primarily due to the absence of crafty designs and geometric constraints for the involved elements. This paper introduces a novel neural rendering method termed Decoupled Hybrid Gaussian Splatting (DHGS), targeting at promoting the rendering quality of novel view synthesis for static driving scenes. The novelty of this work lies in the decoupled and hybrid pixel-level blender for road and non-road layers, without the conventional unified differentiable rendering logic for the entire scene. Still, consistency and continuity in superimposition are preserved through the proposed depth-ordered hybrid rendering strategy. Additionally, an implicit road representation comprised of a Signed Distance Function (SDF) is trained to supervise the road surface with subtle geometric attributes. Accompanied by the use of auxiliary transmittance loss and consistency loss, novel images with imperceptible boundary and elevated fidelity are ultimately obtained. Substantial experiments on the Waymo dataset prove that DHGS outperforms the state-of-the-art methods. The project page where more video evidences are given is: https://ironbrotherstyle.github.io/dhgs_web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06128v2">Gaussian Pancakes: Geometrically-Regularized 3D Gaussian Splatting for Realistic Endoscopic Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Within colorectal cancer diagnostics, conventional colonoscopy techniques face critical limitations, including a limited field of view and a lack of depth information, which can impede the detection of precancerous lesions. Current methods struggle to provide comprehensive and accurate 3D reconstructions of the colonic surface which can help minimize the missing regions and reinspection for pre-cancerous polyps. Addressing this, we introduce 'Gaussian Pancakes', a method that leverages 3D Gaussian Splatting (3D GS) combined with a Recurrent Neural Network-based Simultaneous Localization and Mapping (RNNSLAM) system. By introducing geometric and depth regularization into the 3D GS framework, our approach ensures more accurate alignment of Gaussians with the colon surface, resulting in smoother 3D reconstructions with novel viewing of detailed textures and structures. Evaluations across three diverse datasets show that Gaussian Pancakes enhances novel view synthesis quality, surpassing current leading methods with a 18% boost in PSNR and a 16% improvement in SSIM. It also delivers over 100X faster rendering and more than 10X shorter training times, making it a practical tool for real-time applications. Hence, this holds promise for achieving clinical translation for better detection and diagnosis of colorectal cancer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08723v1">Correspondence-Guided SfM-Free 3D Gaussian Splatting for NVS</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 arXiv admin note: text overlap with arXiv:2312.07504 by other authors
    </div>
    <details class="paper-abstract">
      Novel View Synthesis (NVS) without Structure-from-Motion (SfM) pre-processed camera poses--referred to as SfM-free methods--is crucial for promoting rapid response capabilities and enhancing robustness against variable operating conditions. Recent SfM-free methods have integrated pose optimization, designing end-to-end frameworks for joint camera pose estimation and NVS. However, most existing works rely on per-pixel image loss functions, such as L2 loss. In SfM-free methods, inaccurate initial poses lead to misalignment issue, which, under the constraints of per-pixel image loss functions, results in excessive gradients, causing unstable optimization and poor convergence for NVS. In this study, we propose a correspondence-guided SfM-free 3D Gaussian splatting for NVS. We use correspondences between the target and the rendered result to achieve better pixel alignment, facilitating the optimization of relative poses between frames. We then apply the learned poses to optimize the entire scene. Each 2D screen-space pixel is associated with its corresponding 3D Gaussians through approximated surface rendering to facilitate gradient back propagation. Experimental results underline the superior performance and time efficiency of the proposed approach compared to the state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08524v1">GS-ID: Illumination Decomposition on Gaussian Splatting via Diffusion Prior and Parametric Light Source Optimization</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 15 pages, 13 figures
    </div>
    <details class="paper-abstract">
      We present GS-ID, a novel framework for illumination decomposition on Gaussian Splatting, achieving photorealistic novel view synthesis and intuitive light editing. Illumination decomposition is an ill-posed problem facing three main challenges: 1) priors for geometry and material are often lacking; 2) complex illumination conditions involve multiple unknown light sources; and 3) calculating surface shading with numerous light sources is computationally expensive. To address these challenges, we first introduce intrinsic diffusion priors to estimate the attributes for physically based rendering. Then we divide the illumination into environmental and direct components for joint optimization. Last, we employ deferred rendering to reduce the computational load. Our framework uses a learnable environment map and Spherical Gaussians (SGs) to represent light sources parametrically, therefore enabling controllable and photorealistic relighting on Gaussian Splatting. Extensive experiments and applications demonstrate that GS-ID produces state-of-the-art illumination decomposition results while achieving better geometry reconstruction and rendering performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09875v3">Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-15
      | 💬 8 pages, 7 figures
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel method to supervise 3D Gaussian Splatting (3DGS) scenes using optical tactile sensors. Optical tactile sensors have become widespread in their use in robotics for manipulation and object representation; however, raw optical tactile sensor data is unsuitable to directly supervise a 3DGS scene. Our representation leverages a Gaussian Process Implicit Surface to implicitly represent the object, combining many touches into a unified representation with uncertainty. We merge this model with a monocular depth estimation network, which is aligned in a two stage process, coarsely aligning with a depth camera and then finely adjusting to match our touch data. For every training image, our method produces a corresponding fused depth and uncertainty map. Utilizing this additional information, we propose a new loss function, variance weighted depth supervised loss, for training the 3DGS scene model. We leverage the DenseTact optical tactile sensor and RealSense RGB-D camera to show that combining touch and vision in this manner leads to quantitatively and qualitatively better results than vision or touch alone in a few-view scene syntheses on opaque as well as on reflective and transparent objects. Please see our project page at http://armlabstanford.github.io/touch-gs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08206v1">WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-15
      | 💬 Web: https://water-splatting.github.io
    </div>
    <details class="paper-abstract">
      The underwater 3D scene reconstruction is a challenging, yet interesting problem with applications ranging from naval robots to VR experiences. The problem was successfully tackled by fully volumetric NeRF-based methods which can model both the geometry and the medium (water). Unfortunately, these methods are slow to train and do not offer real-time rendering. More recently, 3D Gaussian Splatting (3DGS) method offered a fast alternative to NeRFs. However, because it is an explicit method that renders only the geometry, it cannot render the medium and is therefore unsuited for underwater reconstruction. Therefore, we propose a novel approach that fuses volumetric rendering with 3DGS to handle underwater data effectively. Our method employs 3DGS for explicit geometry representation and a separate volumetric field (queried once per pixel) for capturing the scattering medium. This dual representation further allows the restoration of the scenes by removing the scattering medium. Our method outperforms state-of-the-art NeRF-based methods in rendering quality on the underwater SeaThru-NeRF dataset. Furthermore, it does so while offering real-time rendering performance, addressing the efficiency limitations of existing methods. Web: https://water-splatting.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04908v2">Dual-Camera Smooth Zoom on Mobile Phones</a></div>
    <div class="paper-meta">
      📅 2024-08-15
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      When zooming between dual cameras on a mobile, noticeable jumps in geometric content and image color occur in the preview, inevitably affecting the user's zoom experience. In this work, we introduce a new task, ie, dual-camera smooth zoom (DCSZ) to achieve a smooth zoom preview. The frame interpolation (FI) technique is a potential solution but struggles with ground-truth collection. To address the issue, we suggest a data factory solution where continuous virtual cameras are assembled to generate DCSZ data by rendering reconstructed 3D models of the scene. In particular, we propose a novel dual-camera smooth zoom Gaussian Splatting (ZoomGS), where a camera-specific encoding is introduced to construct a specific 3D model for each virtual camera. With the proposed data factory, we construct a synthetic dataset for DCSZ, and we utilize it to fine-tune FI models. In addition, we collect real-world dual-zoom images without ground-truth for evaluation. Extensive experiments are conducted with multiple FI methods. The results show that the fine-tuned FI models achieve a significant performance improvement over the original ones on DCSZ task. The datasets, codes, and pre-trained models will are available at https://github.com/ZcsrenlongZ/ZoomGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07595v1">Progressive Radiance Distillation for Inverse Rendering with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-14
    </div>
    <details class="paper-abstract">
      We propose progressive radiance distillation, an inverse rendering method that combines physically-based rendering with Gaussian-based radiance field rendering using a distillation progress map. Taking multi-view images as input, our method starts from a pre-trained radiance field guidance, and distills physically-based light and material parameters from the radiance field using an image-fitting process. The distillation progress map is initialized to a small value, which favors radiance field rendering. During early iterations when fitted light and material parameters are far from convergence, the radiance field fallback ensures the sanity of image loss gradients and avoids local minima that attracts under-fit states. As fitted parameters converge, the physical model gradually takes over and the distillation progress increases correspondingly. In presence of light paths unmodeled by the physical model, the distillation progress never finishes on affected pixels and the learned radiance field stays in the final rendering. With this designed tolerance for physical model limitations, we prevent unmodeled color components from leaking into light and material parameters, alleviating relighting artifacts. Meanwhile, the remaining radiance field compensates for the limitations of the physical model, guaranteeing high-quality novel views synthesis. Experimental results demonstrate that our method significantly outperforms state-of-the-art techniques quality-wise in both novel view synthesis and relighting. The idea of progressive radiance distillation is not limited to Gaussian splatting. We show that it also has positive effects for prominently specular scenes when adapted to a mesh-based inverse rendering method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10683v2">GS-Pose: Generalizable Segmentation-based 6D Object Pose Estimation with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 Project Page: https://dingdingcai.github.io/gs-pose
    </div>
    <details class="paper-abstract">
      This paper introduces GS-Pose, a unified framework for localizing and estimating the 6D pose of novel objects. GS-Pose begins with a set of posed RGB images of a previously unseen object and builds three distinct representations stored in a database. At inference, GS-Pose operates sequentially by locating the object in the input image, estimating its initial 6D pose using a retrieval approach, and refining the pose with a render-and-compare method. The key insight is the application of the appropriate object representation at each stage of the process. In particular, for the refinement step, we leverage 3D Gaussian splatting, a novel differentiable rendering technique that offers high rendering speed and relatively low optimization time. Off-the-shelf toolchains and commodity hardware, such as mobile phones, can be used to capture new objects to be added to the database. Extensive evaluations on the LINEMOD and OnePose-LowTexture datasets demonstrate excellent performance, establishing the new state-of-the-art. Project page: https://dingdingcai.github.io/gs-pose.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07540v1">3D Gaussian Editing with A Single Image</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 10 pages, 12 figures
    </div>
    <details class="paper-abstract">
      The modeling and manipulation of 3D scenes captured from the real world are pivotal in various applications, attracting growing research interest. While previous works on editing have achieved interesting results through manipulating 3D meshes, they often require accurately reconstructed meshes to perform editing, which limits their application in 3D content generation. To address this gap, we introduce a novel single-image-driven 3D scene editing approach based on 3D Gaussian Splatting, enabling intuitive manipulation via directly editing the content on a 2D image plane. Our method learns to optimize the 3D Gaussians to align with an edited version of the image rendered from a user-specified viewpoint of the original scene. To capture long-range object deformation, we introduce positional loss into the optimization process of 3D Gaussian Splatting and enable gradient propagation through reparameterization. To handle occluded 3D Gaussians when rendering from the specified viewpoint, we build an anchor-based structure and employ a coarse-to-fine optimization strategy capable of handling long-range deformation while maintaining structural stability. Furthermore, we design a novel masking strategy to adaptively identify non-rigid deformation regions for fine-scale modeling. Extensive experiments show the effectiveness of our method in handling geometric details, long-range, and non-rigid deformation, demonstrating superior editing flexibility and quality compared to previous approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09717v3">From NeRFs to Gaussian Splats, and Back</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      For robotics applications where there is a limited number of (typically ego-centric) views, parametric representations such as neural radiance fields (NeRFs) generalize better than non-parametric ones such as Gaussian splatting (GS) to views that are very different from those in the training data; GS however can render much faster than NeRFs. We develop a procedure to convert back and forth between the two. Our approach achieves the best of both NeRFs (superior PSNR, SSIM, and LPIPS on dissimilar views, and a compact representation) and GS (real-time rendering and ability for easily modifying the representation); the computational cost of these conversions is minor compared to training the two from scratch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.02902v2">HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-13
      | 💬 accepted to ECCV 2024
    </div>
    <details class="paper-abstract">
      3D head animation has seen major quality and runtime improvements over the last few years, particularly empowered by the advances in differentiable rendering and neural radiance fields. Real-time rendering is a highly desirable goal for real-world applications. We propose HeadGaS, a model that uses 3D Gaussian Splats (3DGS) for 3D head reconstruction and animation. In this paper we introduce a hybrid model that extends the explicit 3DGS representation with a base of learnable latent features, which can be linearly blended with low-dimensional parameters from parametric head models to obtain expression-dependent color and opacity values. We demonstrate that HeadGaS delivers state-of-the-art results in real-time inference frame rates, surpassing baselines by up to 2dB, while accelerating rendering speed by over x10.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06975v1">SpectralGaussians: Semantic, spectral 3D Gaussian splatting for multi-spectral scene representation, visualization and analysis</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      We propose a novel cross-spectral rendering framework based on 3D Gaussian Splatting (3DGS) that generates realistic and semantically meaningful splats from registered multi-view spectrum and segmentation maps. This extension enhances the representation of scenes with multiple spectra, providing insights into the underlying materials and segmentation. We introduce an improved physically-based rendering approach for Gaussian splats, estimating reflectance and lights per spectra, thereby enhancing accuracy and realism. In a comprehensive quantitative and qualitative evaluation, we demonstrate the superior performance of our approach with respect to other recent learning-based spectral scene representation approaches (i.e., XNeRF and SpectralNeRF) as well as other non-spectral state-of-the-art learning-based approaches. Our work also demonstrates the potential of spectral scene understanding for precise scene editing techniques like style transfer, inpainting, and removal. Thereby, our contributions address challenges in multi-spectral scene representation, rendering, and editing, offering new possibilities for diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05220v2">StylizedGS: Controllable Stylization for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      As XR technology continues to advance rapidly, 3D generation and editing are increasingly crucial. Among these, stylization plays a key role in enhancing the appearance of 3D models. By utilizing stylization, users can achieve consistent artistic effects in 3D editing using a single reference style image, making it a user-friendly editing method. However, recent NeRF-based 3D stylization methods encounter efficiency issues that impact the user experience, and their implicit nature limits their ability to accurately transfer geometric pattern styles. Additionally, the ability for artists to apply flexible control over stylized scenes is considered highly desirable to foster an environment conducive to creative exploration. To address the above issues, we introduce StylizedGS, an efficient 3D neural style transfer framework with adaptable control over perceptual factors based on 3D Gaussian Splatting (3DGS) representation. We propose a filter-based refinement to eliminate floaters that affect the stylization effects in the scene reconstruction process. The nearest neighbor-based style loss is introduced to achieve stylization by fine-tuning the geometry and color parameters of 3DGS, while a depth preservation loss with other regularizations is proposed to prevent the tampering of geometry content. Moreover, facilitated by specially designed losses, StylizedGS enables users to control color, stylized scale, and regions during the stylization to possess customization capabilities. Our method achieves high-quality stylization results characterized by faithful brushstrokes and geometric consistency with flexible controls. Extensive experiments across various scenes and styles demonstrate the effectiveness and efficiency of our method concerning both stylization quality and inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06286v1">Mipmap-GS: Let Gaussians Deform with Scale-specific Mipmap for Anti-aliasing Rendering</a></div>
    <div class="paper-meta">
      📅 2024-08-12
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has attracted great attention in novel view synthesis because of its superior rendering efficiency and high fidelity. However, the trained Gaussians suffer from severe zooming degradation due to non-adjustable representation derived from single-scale training. Though some methods attempt to tackle this problem via post-processing techniques such as selective rendering or filtering techniques towards primitives, the scale-specific information is not involved in Gaussians. In this paper, we propose a unified optimization method to make Gaussians adaptive for arbitrary scales by self-adjusting the primitive properties (e.g., color, shape and size) and distribution (e.g., position). Inspired by the mipmap technique, we design pseudo ground-truth for the target scale and propose a scale-consistency guidance loss to inject scale information into 3D Gaussians. Our method is a plug-in module, applicable for any 3DGS models to solve the zoom-in and zoom-out aliasing. Extensive experiments demonstrate the effectiveness of our method. Notably, our method outperforms 3DGS in PSNR by an average of 9.25 dB for zoom-in and 10.40 dB for zoom-out on the NeRF Synthetic dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06030v1">Developing Smart MAVs for Autonomous Inspection in GPS-denied Constructions</a></div>
    <div class="paper-meta">
      📅 2024-08-12
    </div>
    <details class="paper-abstract">
      Smart Micro Aerial Vehicles (MAVs) have transformed infrastructure inspection by enabling efficient, high-resolution monitoring at various stages of construction, including hard-to-reach areas. Traditional manual operation of drones in GPS-denied environments, such as industrial facilities and infrastructure, is labour-intensive, tedious and prone to error. This study presents an innovative framework for smart MAV inspections in such complex and GPS-denied indoor environments. The framework features a hierarchical perception and planning system that identifies regions of interest and optimises task paths. It also presents an advanced MAV system with enhanced localisation and motion planning capabilities, integrated with Neural Reconstruction technology for comprehensive 3D reconstruction of building structures. The effectiveness of the framework was empirically validated in a 4,000 square meters indoor infrastructure facility with an interior length of 80 metres, a width of 50 metres and a height of 7 metres. The main structure consists of columns and walls. Experimental results show that our MAV system performs exceptionally well in autonomous inspection tasks, achieving a 100\% success rate in generating and executing scan paths. Extensive experiments validate the manoeuvrability of our developed MAV, achieving a 100\% success rate in motion planning with a tracking error of less than 0.1 metres. In addition, the enhanced reconstruction method using 3D Gaussian Splatting technology enables the generation of high-fidelity rendering models from the acquired data. Overall, our novel method represents a significant advancement in the use of robotics for infrastructure inspection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14037v3">GaussianTalker: Speaker-specific Talking Head Synthesis via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-09
      | 💬 Accepted by ACM MM 2024. Project page: https://yuhongyun777.github.io/GaussianTalker/
    </div>
    <details class="paper-abstract">
      Recent works on audio-driven talking head synthesis using Neural Radiance Fields (NeRF) have achieved impressive results. However, due to inadequate pose and expression control caused by NeRF implicit representation, these methods still have some limitations, such as unsynchronized or unnatural lip movements, and visual jitter and artifacts. In this paper, we propose GaussianTalker, a novel method for audio-driven talking head synthesis based on 3D Gaussian Splatting. With the explicit representation property of 3D Gaussians, intuitive control of the facial motion is achieved by binding Gaussians to 3D facial models. GaussianTalker consists of two modules, Speaker-specific Motion Translator and Dynamic Gaussian Renderer. Speaker-specific Motion Translator achieves accurate lip movements specific to the target speaker through universalized audio feature extraction and customized lip motion generation. Dynamic Gaussian Renderer introduces Speaker-specific BlendShapes to enhance facial detail representation via a latent pose, delivering stable and realistic rendered videos. Extensive experimental results suggest that GaussianTalker outperforms existing state-of-the-art methods in talking head synthesis, delivering precise lip synchronization and exceptional visual quality. Our method achieves rendering speeds of 130 FPS on NVIDIA RTX4090 GPU, significantly exceeding the threshold for real-time rendering performance, and can potentially be deployed on other hardware platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04426v1">A Review of 3D Reconstruction Techniques for Deformable Tissues in Robotic Surgery</a></div>
    <div class="paper-meta">
      📅 2024-08-08
      | 💬 To appear in MICCAI 2024 EARTH Workshop. Code availability: https://github.com/Epsilon404/surgicalnerf
    </div>
    <details class="paper-abstract">
      As a crucial and intricate task in robotic minimally invasive surgery, reconstructing surgical scenes using stereo or monocular endoscopic video holds immense potential for clinical applications. NeRF-based techniques have recently garnered attention for the ability to reconstruct scenes implicitly. On the other hand, Gaussian splatting-based 3D-GS represents scenes explicitly using 3D Gaussians and projects them onto a 2D plane as a replacement for the complex volume rendering in NeRF. However, these methods face challenges regarding surgical scene reconstruction, such as slow inference, dynamic scenes, and surgical tool occlusion. This work explores and reviews state-of-the-art (SOTA) approaches, discussing their innovations and implementation principles. Furthermore, we replicate the models and conduct testing and evaluation on two datasets. The test results demonstrate that with advancements in these techniques, achieving real-time, high-quality reconstructions becomes feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08759v2">GaussianForest: Hierarchical-Hybrid 3D Gaussian Splatting for Compressed Scene Modeling</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      The field of novel-view synthesis has recently witnessed the emergence of 3D Gaussian Splatting, which represents scenes in a point-based manner and renders through rasterization. This methodology, in contrast to Radiance Fields that rely on ray tracing, demonstrates superior rendering quality and speed. However, the explicit and unstructured nature of 3D Gaussians poses a significant storage challenge, impeding its broader application. To address this challenge, we introduce the Gaussian-Forest modeling framework, which hierarchically represents a scene as a forest of hybrid 3D Gaussians. Each hybrid Gaussian retains its unique explicit attributes while sharing implicit ones with its sibling Gaussians, thus optimizing parameterization with significantly fewer variables. Moreover, adaptive growth and pruning strategies are designed, ensuring detailed representation in complex regions and a notable reduction in the number of required Gaussians. Extensive experiments demonstrate that Gaussian-Forest not only maintains comparable speed and quality but also achieves a compression rate surpassing 10 times, marking a significant advancement in efficient scene modeling. Codes will be available at https://github.com/Xian-Bei/GaussianForest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16043v2">Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing</a></div>
    <div class="paper-meta">
      📅 2024-08-08
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel differentiable point-based rendering framework to achieve photo-realistic relighting. To make the reconstructed scene relightable, we enhance vanilla 3D Gaussians by associating extra properties, including normal vectors, BRDF parameters, and incident lighting from various directions. From a collection of multi-view images, the 3D scene is optimized through 3D Gaussian Splatting while BRDF and lighting are decomposed by physically based differentiable rendering. To produce plausible shadow effects in photo-realistic relighting, we introduce an innovative point-based ray tracing with the bounding volume hierarchies for efficient visibility pre-computation. Extensive experiments demonstrate our improved BRDF estimation, novel view synthesis and relighting results compared to state-of-the-art approaches. The proposed framework showcases the potential to revolutionize the mesh-based graphics pipeline with a point-based pipeline enabling editing, tracing, and relighting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03825v1">Towards Real-Time Gaussian Splatting: Accelerating 3DGS through Photometric SLAM</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 This extended abstract has been submitted to be presented at an IEEE conference. It will be made available online by IEEE but will not be published in IEEE Xplore
    </div>
    <details class="paper-abstract">
      Initial applications of 3D Gaussian Splatting (3DGS) in Visual Simultaneous Localization and Mapping (VSLAM) demonstrate the generation of high-quality volumetric reconstructions from monocular video streams. However, despite these promising advancements, current 3DGS integrations have reduced tracking performance and lower operating speeds compared to traditional VSLAM. To address these issues, we propose integrating 3DGS with Direct Sparse Odometry, a monocular photometric SLAM system. We have done preliminary experiments showing that using Direct Sparse Odometry point cloud outputs, as opposed to standard structure-from-motion methods, significantly shortens the training time needed to achieve high-quality renders. Reducing 3DGS training time enables the development of 3DGS-integrated SLAM systems that operate in real-time on mobile hardware. These promising initial findings suggest further exploration is warranted in combining traditional VSLAM systems with 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03822v1">Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 Project page: https://maincold2.github.io/c3dgs/
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has recently emerged as an alternative representation that leverages a 3D Gaussian-based representation and introduces an approximated volumetric rendering, achieving very fast rendering speed and promising image quality. Furthermore, subsequent studies have successfully extended 3DGS to dynamic 3D scenes, demonstrating its wide range of applications. However, a significant drawback arises as 3DGS and its following methods entail a substantial number of Gaussians to maintain the high fidelity of the rendered images, which requires a large amount of memory and storage. To address this critical issue, we place a specific emphasis on two key objectives: reducing the number of Gaussian points without sacrificing performance and compressing the Gaussian attributes, such as view-dependent color and covariance. To this end, we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance. In addition, we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics. Finally, we learn codebooks to compactly represent the geometric and temporal attributes by residual vector quantization. With model compression techniques such as quantization and entropy coding, we consistently show over 25x reduced storage and enhanced rendering speed compared to 3DGS for static scenes, while maintaining the quality of the scene representation. For dynamic scenes, our approach achieves more than 12x storage efficiency and retains a high-quality reconstruction compared to the existing state-of-the-art methods. Our work provides a comprehensive framework for 3D scene representation, achieving high performance, fast training, compactness, and real-time rendering. Our project page is available at https://maincold2.github.io/c3dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03753v1">3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 The 18th European Conference on Computer Vision ECCV 2024
    </div>
    <details class="paper-abstract">
      The use of 3D Gaussians as representation of radiance fields has enabled high quality novel view synthesis at real-time rendering speed. However, the choice of optimising the outgoing radiance of each Gaussian independently as spherical harmonics results in unsatisfactory view dependent effects. In response to these limitations, our work, Factorised Tensorial Illumination for 3D Gaussian Splatting, or 3iGS, improves upon 3D Gaussian Splatting (3DGS) rendering quality. Instead of optimising a single outgoing radiance parameter, 3iGS enhances 3DGS view-dependent effects by expressing the outgoing radiance as a function of a local illumination field and Bidirectional Reflectance Distribution Function (BRDF) features. We optimise a continuous incident illumination field through a Tensorial Factorisation representation, while separately fine-tuning the BRDF features of each 3D Gaussian relative to this illumination field. Our methodology significantly enhances the rendering quality of specular view-dependent effects of 3DGS, while maintaining rapid training and rendering speeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01126v2">IG-SLAM: Instant Gaussian SLAM</a></div>
    <div class="paper-meta">
      📅 2024-08-07
      | 💬 8 pages, 3 page ref, 5 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has recently shown promising results as an alternative scene representation in SLAM systems to neural implicit representations. However, current methods either lack dense depth maps to supervise the mapping process or detailed training designs that consider the scale of the environment. To address these drawbacks, we present IG-SLAM, a dense RGB-only SLAM system that employs robust Dense-SLAM methods for tracking and combines them with Gaussian Splatting. A 3D map of the environment is constructed using accurate pose and dense depth provided by tracking. Additionally, we utilize depth uncertainty in map optimization to improve 3D reconstruction. Our decay strategy in map optimization enhances convergence and allows the system to run at 10 fps in a single process. We demonstrate competitive performance with state-of-the-art RGB-only SLAM systems while achieving faster operation speeds. We present our experiments on the Replica, TUM-RGBD, ScanNet, and EuRoC datasets. The system achieves photo-realistic 3D reconstruction in large-scale sequences, particularly in the EuRoC dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03538v1">PRTGS: Precomputed Radiance Transfer of Gaussian Splats for Real-Time High-Quality Relighting</a></div>
    <div class="paper-meta">
      📅 2024-08-07
    </div>
    <details class="paper-abstract">
      We proposed Precomputed RadianceTransfer of GaussianSplats (PRTGS), a real-time high-quality relighting method for Gaussian splats in low-frequency lighting environments that captures soft shadows and interreflections by precomputing 3D Gaussian splats' radiance transfer. Existing studies have demonstrated that 3D Gaussian splatting (3DGS) outperforms neural fields' efficiency for dynamic lighting scenarios. However, the current relighting method based on 3DGS still struggles to compute high-quality shadow and indirect illumination in real time for dynamic light, leading to unrealistic rendering results. We solve this problem by precomputing the expensive transport simulations required for complex transfer functions like shadowing, the resulting transfer functions are represented as dense sets of vectors or matrices for every Gaussian splat. We introduce distinct precomputing methods tailored for training and rendering stages, along with unique ray tracing and indirect lighting precomputation techniques for 3D Gaussian splats to accelerate training speed and compute accurate indirect lighting related to environment light. Experimental analyses demonstrate that our approach achieves state-of-the-art visual quality while maintaining competitive training times and allows high-quality real-time (30+ fps) relighting for dynamic light and relatively complex scenes at 1080p resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07950v3">Reinforcement Learning with Generalizable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-06
      | 💬 7 pages,2 figures
    </div>
    <details class="paper-abstract">
      An excellent representation is crucial for reinforcement learning (RL) performance, especially in vision-based reinforcement learning tasks. The quality of the environment representation directly influences the achievement of the learning task. Previous vision-based RL typically uses explicit or implicit ways to represent environments, such as images, points, voxels, and neural radiance fields. However, these representations contain several drawbacks. They cannot either describe complex local geometries or generalize well to unseen scenes, or require precise foreground masks. Moreover, these implicit neural representations are akin to a ``black box", significantly hindering interpretability. 3D Gaussian Splatting (3DGS), with its explicit scene representation and differentiable rendering nature, is considered a revolutionary change for reconstruction and representation methods. In this paper, we propose a novel Generalizable Gaussian Splatting framework to be the representation of RL tasks, called GSRL. Through validation in the RoboMimic environment, our method achieves better results than other baselines in multiple tasks, improving the performance by 10%, 44%, and 15% compared with baselines on the hardest task. This work is the first attempt to leverage generalizable 3DGS as a representation for RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01269v1">A General Framework to Boost 3D GS Initialization for Text-to-3D Generation by Lexical Richness</a></div>
    <div class="paper-meta">
      📅 2024-08-02
    </div>
    <details class="paper-abstract">
      Text-to-3D content creation has recently received much attention, especially with the prevalence of 3D Gaussians Splatting. In general, GS-based methods comprise two key stages: initialization and rendering optimization. To achieve initialization, existing works directly apply random sphere initialization or 3D diffusion models, e.g., Point-E, to derive the initial shapes. However, such strategies suffer from two critical yet challenging problems: 1) the final shapes are still similar to the initial ones even after training; 2) shapes can be produced only from simple texts, e.g., "a dog", not for lexically richer texts, e.g., "a dog is sitting on the top of the airplane". To address these problems, this paper proposes a novel general framework to boost the 3D GS Initialization for text-to-3D generation upon the lexical richness. Our key idea is to aggregate 3D Gaussians into spatially uniform voxels to represent complex shapes while enabling the spatial interaction among the 3D Gaussians and semantic interaction between Gaussians and texts. Specifically, we first construct a voxelized representation, where each voxel holds a 3D Gaussian with its position, scale, and rotation fixed while setting opacity as the sole factor to determine a position's occupancy. We then design an initialization network mainly consisting of two novel components: 1) Global Information Perception (GIP) block and 2) Gaussians-Text Fusion (GTF) block. Such a design enables each 3D Gaussian to assimilate the spatial information from other areas and semantic information from texts. Extensive experiments show the superiority of our framework of high-quality 3D GS initialization against the existing methods, e.g., Shap-E, by taking lexically simple, medium, and hard texts. Also, our framework can be seamlessly plugged into SoTA training frameworks, e.g., LucidDreamer, for semantically consistent text-to-3D generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01225v1">Reality Fusion: Robust Real-time Immersive Mobile Robot Teleoperation with Volumetric Visual Data Fusion</a></div>
    <div class="paper-meta">
      📅 2024-08-02
      | 💬 Accepted, to appear at IROS 2024
    </div>
    <details class="paper-abstract">
      We introduce Reality Fusion, a novel robot teleoperation system that localizes, streams, projects, and merges a typical onboard depth sensor with a photorealistic, high resolution, high framerate, and wide field of view (FoV) rendering of the complex remote environment represented as 3D Gaussian splats (3DGS). Our framework enables robust egocentric and exocentric robot teleoperation in immersive VR, with the 3DGS effectively extending spatial information of a depth sensor with limited FoV and balancing the trade-off between data streaming costs and data visual quality. We evaluated our framework through a user study with 24 participants, which revealed that Reality Fusion leads to significantly better user performance, situation awareness, and user preferences. To support further research and development, we provide an open-source implementation with an easy-to-replicate custom-made telepresence robot, a high-performance virtual reality 3DGS renderer, and an immersive robot control package. (Source code: https://github.com/uhhhci/RealityFusion)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00254v1">LoopSparseGS: Loop Based Sparse-View Friendly Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-08-01
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Despite the photorealistic novel view synthesis (NVS) performance achieved by the original 3D Gaussian splatting (3DGS), its rendering quality significantly degrades with sparse input views. This performance drop is mainly caused by the limited number of initial points generated from the sparse input, insufficient supervision during the training process, and inadequate regularization of the oversized Gaussian ellipsoids. To handle these issues, we propose the LoopSparseGS, a loop-based 3DGS framework for the sparse novel view synthesis task. In specific, we propose a loop-based Progressive Gaussian Initialization (PGI) strategy that could iteratively densify the initialized point cloud using the rendered pseudo images during the training process. Then, the sparse and reliable depth from the Structure from Motion, and the window-based dense monocular depth are leveraged to provide precise geometric supervision via the proposed Depth-alignment Regularization (DAR). Additionally, we introduce a novel Sparse-friendly Sampling (SFS) strategy to handle oversized Gaussian ellipsoids leading to large pixel errors. Comprehensive experiments on four datasets demonstrate that LoopSparseGS outperforms existing state-of-the-art methods for sparse-input novel view synthesis, across indoor, outdoor, and object-level scenes with various image resolutions.
    </details>
</div>
