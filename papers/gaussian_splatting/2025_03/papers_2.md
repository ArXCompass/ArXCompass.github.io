# gaussian splatting - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12507v2">3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 Our paper has been accepted by CVPR 2025. For more details and updates, please visit our project website: https://research.nvidia.com/labs/toronto-ai/3DGUT
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables efficient reconstruction and high-fidelity real-time rendering of complex scenes on consumer hardware. However, due to its rasterization-based formulation, 3DGS is constrained to ideal pinhole cameras and lacks support for secondary lighting effects. Recent methods address these limitations by tracing the particles instead, but, this comes at the cost of significantly slower rendering. In this work, we propose 3D Gaussian Unscented Transform (3DGUT), replacing the EWA splatting formulation with the Unscented Transform that approximates the particles through sigma points, which can be projected exactly under any nonlinear projection function. This modification enables trivial support of distorted cameras with time dependent effects such as rolling shutter, while retaining the efficiency of rasterization. Additionally, we align our rendering formulation with that of tracing-based methods, enabling secondary ray tracing required to represent phenomena such as reflections and refraction within the same 3D representation. The source code is available at: https://github.com/nv-tlabs/3dgrut.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10219v3">PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 CVPR 2025, Project Page: https://pup3dgs.github.io/
    </div>
    <details class="paper-abstract">
      Recent advances in novel view synthesis have enabled real-time rendering speeds with high reconstruction accuracy. 3D Gaussian Splatting (3D-GS), a foundational point-based parametric 3D scene representation, models scenes as large sets of 3D Gaussians. However, complex scenes can consist of millions of Gaussians, resulting in high storage and memory requirements that limit the viability of 3D-GS on devices with limited resources. Current techniques for compressing these pretrained models by pruning Gaussians rely on combining heuristics to determine which Gaussians to remove. At high compression ratios, these pruned scenes suffer from heavy degradation of visual fidelity and loss of foreground details. In this paper, we propose a principled sensitivity pruning score that preserves visual fidelity and foreground details at significantly higher compression ratios than existing approaches. It is computed as a second-order approximation of the reconstruction error on the training views with respect to the spatial parameters of each Gaussian. Additionally, we propose a multi-round prune-refine pipeline that can be applied to any pretrained 3D-GS model without changing its training pipeline. After pruning 90% of Gaussians, a substantially higher percentage than previous methods, our PUP 3D-GS pipeline increases average rendering speed by 3.56$\times$ while retaining more salient foreground information and achieving higher image quality metrics than existing techniques on scenes from the Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18794v1">NexusGS: Sparse View Synthesis with Epipolar Depth Priors in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 This paper is accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have noticeably advanced photo-realistic novel view synthesis using images from densely spaced camera viewpoints. However, these methods struggle in few-shot scenarios due to limited supervision. In this paper, we present NexusGS, a 3DGS-based approach that enhances novel view synthesis from sparse-view images by directly embedding depth information into point clouds, without relying on complex manual regularizations. Exploiting the inherent epipolar geometry of 3DGS, our method introduces a novel point cloud densification strategy that initializes 3DGS with a dense point cloud, reducing randomness in point placement while preventing over-smoothing and overfitting. Specifically, NexusGS comprises three key steps: Epipolar Depth Nexus, Flow-Resilient Depth Blending, and Flow-Filtered Depth Pruning. These steps leverage optical flow and camera poses to compute accurate depth maps, while mitigating the inaccuracies often associated with optical flow. By incorporating epipolar depth priors, NexusGS ensures reliable dense point cloud coverage and supports stable 3DGS training under sparse-view conditions. Experiments demonstrate that NexusGS significantly enhances depth accuracy and rendering quality, surpassing state-of-the-art methods by a considerable margin. Furthermore, we validate the superiority of our generated point clouds by substantially boosting the performance of competing methods. Project page: https://usmizuki.github.io/NexusGS/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17811v2">Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 CVPR 2025. Project page here: https://gaoxiangjun.github.io/mani_gs/
    </div>
    <details class="paper-abstract">
      Neural 3D representations such as Neural Radiance Fields (NeRF), excel at producing photo-realistic rendering results but lack the flexibility for manipulation and editing which is crucial for content creation. Previous works have attempted to address this issue by deforming a NeRF in canonical space or manipulating the radiance field based on an explicit mesh. However, manipulating NeRF is not highly controllable and requires a long training and inference time. With the emergence of 3D Gaussian Splatting (3DGS), extremely high-fidelity novel view synthesis can be achieved using an explicit point-based 3D representation with much faster training and rendering speed. However, there is still a lack of effective means to manipulate 3DGS freely while maintaining rendering quality. In this work, we aim to tackle the challenge of achieving manipulable photo-realistic rendering. We propose to utilize a triangular mesh to manipulate 3DGS directly with self-adaptation. This approach reduces the need to design various algorithms for different types of Gaussian manipulation. By utilizing a triangle shape-aware Gaussian binding and adapting method, we can achieve 3DGS manipulation and preserve high-fidelity rendering after manipulation. Our approach is capable of handling large deformations, local manipulations, and soft body simulations while keeping high-quality rendering. Furthermore, we demonstrate that our method is also effective with inaccurate meshes extracted from 3DGS. Experiments conducted demonstrate the effectiveness of our method and its superiority over baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18718v1">GS-Marker: Generalizable and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In the Generative AI era, safeguarding 3D models has become increasingly urgent. While invisible watermarking is well-established for 2D images with encoder-decoder frameworks, generalizable and robust solutions for 3D remain elusive. The main difficulty arises from the renderer between the 3D encoder and 2D decoder, which disrupts direct gradient flow and complicates training. Existing 3D methods typically rely on per-scene iterative optimization, resulting in time inefficiency and limited generalization. In this work, we propose a single-pass watermarking approach for 3D Gaussian Splatting (3DGS), a well-known yet underexplored representation for watermarking. We identify two major challenges: (1) ensuring effective training generalized across diverse 3D models, and (2) reliably extracting watermarks from free-view renderings, even under distortions. Our framework, named GS-Marker, incorporates a 3D encoder to embed messages, distortion layers to enhance resilience against various distortions, and a 2D decoder to extract watermarks from renderings. A key innovation is the Adaptive Marker Control mechanism that adaptively perturbs the initially optimized 3DGS, escaping local minima and improving both training stability and convergence. Extensive experiments show that GS-Marker outperforms per-scene training approaches in terms of decoding accuracy and model fidelity, while also significantly reducing computation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08107v3">IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 Code Page: https://github.com/wu-cvgl/IncEventGS
    </div>
    <details class="paper-abstract">
      Implicit neural representation and explicit 3D Gaussian Splatting (3D-GS) for novel view synthesis have achieved remarkable progress with frame-based camera (e.g. RGB and RGB-D cameras) recently. Compared to frame-based camera, a novel type of bio-inspired visual sensor, i.e. event camera, has demonstrated advantages in high temporal resolution, high dynamic range, low power consumption and low latency. Due to its unique asynchronous and irregular data capturing process, limited work has been proposed to apply neural representation or 3D Gaussian splatting for an event camera. In this work, we present IncEventGS, an incremental 3D Gaussian Splatting reconstruction algorithm with a single event camera. To recover the 3D scene representation incrementally, we exploit the tracking and mapping paradigm of conventional SLAM pipelines for IncEventGS. Given the incoming event stream, the tracker firstly estimates an initial camera motion based on prior reconstructed 3D-GS scene representation. The mapper then jointly refines both the 3D scene representation and camera motion based on the previously estimated motion trajectory from the tracker. The experimental results demonstrate that IncEventGS delivers superior performance compared to prior NeRF-based methods and other related baselines, even we do not have the ground-truth camera poses. Furthermore, our method can also deliver better performance compared to state-of-the-art event visual odometry methods in terms of camera motion estimation. Code is publicly available at: https://github.com/wu-cvgl/IncEventGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18682v1">Hardware-Rasterized Ray-Based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      We present a novel, hardware rasterized rendering approach for ray-based 3D Gaussian Splatting (RayGS), obtaining both fast and high-quality results for novel view synthesis. Our work contains a mathematically rigorous and geometrically intuitive derivation about how to efficiently estimate all relevant quantities for rendering RayGS models, structured with respect to standard hardware rasterization shaders. Our solution is the first enabling rendering RayGS models at sufficiently high frame rates to support quality-sensitive applications like Virtual and Mixed Reality. Our second contribution enables alias-free rendering for RayGS, by addressing MIP-related issues arising when rendering diverging scales during training and testing. We demonstrate significant performance gains, across different benchmark scenes, while retaining state-of-the-art appearance quality of RayGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18640v1">LLGS: Unsupervised Gaussian Splatting for Image Enhancement and Reconstruction in Pure Dark Environment</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has shown remarkable capabilities in novel view rendering tasks and exhibits significant potential for multi-view optimization.However, the original 3D Gaussian Splatting lacks color representation for inputs in low-light environments. Simply using enhanced images as inputs would lead to issues with multi-view consistency, and current single-view enhancement systems rely on pre-trained data, lacking scene generalization. These problems limit the application of 3D Gaussian Splatting in low-light conditions in the field of robotics, including high-fidelity modeling and feature matching. To address these challenges, we propose an unsupervised multi-view stereoscopic system based on Gaussian Splatting, called Low-Light Gaussian Splatting (LLGS). This system aims to enhance images in low-light environments while reconstructing the scene. Our method introduces a decomposable Gaussian representation called M-Color, which separately characterizes color information for targeted enhancement. Furthermore, we propose an unsupervised optimization method with zero-knowledge priors, using direction-based enhancement to ensure multi-view consistency. Experiments conducted on real-world datasets demonstrate that our system outperforms state-of-the-art methods in both low-light enhancement and 3D Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12836v2">CompMarkGS: Robust Watermarking for Compression 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 23 pages, 17 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables rapid differentiable rendering for 3D reconstruction and novel view synthesis, leading to its widespread commercial use. Consequently, copyright protection via watermarking has become critical. However, because 3DGS relies on millions of Gaussians, which require gigabytes of storage, efficient transfer and storage require compression. Existing 3DGS watermarking methods are vulnerable to quantization-based compression, often resulting in the loss of the embedded watermark. To address this challenge, we propose a novel watermarking method that ensures watermark robustness after model compression while maintaining high rendering quality. In detail, we incorporate a quantization distortion layer that simulates compression during training, preserving the watermark under quantization-based compression. Also, we propose a learnable watermark embedding feature that embeds the watermark into the anchor feature, ensuring structural consistency and seamless integration into the 3D scene. Furthermore, we present a frequency-aware anchor growing mechanism to enhance image quality in high-frequency regions by effectively identifying Guassians within these regions. Experimental results confirm that our method preserves the watermark and maintains superior image quality under high compression, validating it as a promising approach for a secure 3DGS model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18458v1">StableGS: A Floater-Free Framework for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Recent years have witnessed remarkable success of 3D Gaussian Splatting (3DGS) in novel view synthesis, surpassing prior differentiable rendering methods in both quality and efficiency. However, its training process suffers from coupled opacity-color optimization that frequently converges to local minima, producing floater artifacts that degrade visual fidelity. We present StableGS, a framework that eliminates floaters through cross-view depth consistency constraints while introducing a dual-opacity GS model to decouple geometry and material properties of translucent objects. To further enhance reconstruction quality in weakly-textured regions, we integrate DUSt3R depth estimation, significantly improving geometric stability. Our method fundamentally addresses 3DGS training instabilities, outperforming existing state-of-the-art methods across open-source datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18421v1">4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 CVPR2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has substantial potential for enabling photorealistic Free-Viewpoint Video (FVV) experiences. However, the vast number of Gaussians and their associated attributes poses significant challenges for storage and transmission. Existing methods typically handle dynamic 3DGS representation and compression separately, neglecting motion information and the rate-distortion (RD) trade-off during training, leading to performance degradation and increased model redundancy. To address this gap, we propose 4DGC, a novel rate-aware 4D Gaussian compression framework that significantly reduces storage size while maintaining superior RD performance for FVV. Specifically, 4DGC introduces a motion-aware dynamic Gaussian representation that utilizes a compact motion grid combined with sparse compensated Gaussians to exploit inter-frame similarities. This representation effectively handles large motions, preserving quality and reducing temporal redundancy. Furthermore, we present an end-to-end compression scheme that employs differentiable quantization and a tiny implicit entropy model to compress the motion grid and compensated Gaussians efficiently. The entire framework is jointly optimized using a rate-distortion trade-off. Extensive experiments demonstrate that 4DGC supports variable bitrates and consistently outperforms existing methods in RD performance across multiple datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03714v3">MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 CVPR2025 (camera ready ver.). The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/MoDecGS-site/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in scene representation and neural rendering, with intense efforts focused on adapting it for dynamic scenes. Despite delivering remarkable rendering quality and speed, existing methods struggle with storage demands and representing complex real-world motions. To tackle these issues, we propose MoDecGS, a memory-efficient Gaussian splatting framework designed for reconstructing novel views in challenging scenarios with complex motions. We introduce GlobaltoLocal Motion Decomposition (GLMD) to effectively capture dynamic motions in a coarsetofine manner. This approach leverages Global Canonical Scaffolds (Global CS) and Local Canonical Scaffolds (Local CS), extending static Scaffold representation to dynamic video reconstruction. For Global CS, we propose Global Anchor Deformation (GAD) to efficiently represent global dynamics along complex motions, by directly deforming the implicit Scaffold attributes which are anchor position, offset, and local context features. Next, we finely adjust local motions via the Local Gaussian Deformation (LGD) of Local CS explicitly. Additionally, we introduce Temporal Interval Adjustment (TIA) to automatically control the temporal coverage of each Local CS during training, allowing MoDecGS to find optimal interval assignments based on the specified number of temporal segments. Extensive evaluations demonstrate that MoDecGS achieves an average 70% reduction in model size over stateoftheart methods for dynamic 3D Gaussians from realworld dynamic videos while maintaining or even improving rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08317v3">Uni-Gaussians: Unifying Camera and Lidar Simulation with Gaussians for Dynamic Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles necessitates comprehensive simulation of multi-sensor data, encompassing inputs from both cameras and LiDAR sensors, across various dynamic driving scenarios. Neural rendering techniques, which utilize collected raw sensor data to simulate these dynamic environments, have emerged as a leading methodology. While NeRF-based approaches can uniformly represent scenes for rendering data from both camera and LiDAR, they are hindered by slow rendering speeds due to dense sampling. Conversely, Gaussian Splatting-based methods employ Gaussian primitives for scene representation and achieve rapid rendering through rasterization. However, these rasterization-based techniques struggle to accurately model non-linear optical sensors. This limitation restricts their applicability to sensors beyond pinhole cameras. To address these challenges and enable unified representation of dynamic driving scenarios using Gaussian primitives, this study proposes a novel hybrid approach. Our method utilizes rasterization for rendering image data while employing Gaussian ray-tracing for LiDAR data rendering. Experimental results on public datasets demonstrate that our approach outperforms current state-of-the-art methods. This work presents a unified and efficient solution for realistic simulation of camera and LiDAR data in autonomous driving scenarios using Gaussian primitives, offering significant advancements in both rendering quality and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18640v1">LLGS: Unsupervised Gaussian Splatting for Image Enhancement and Reconstruction in Pure Dark Environment</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has shown remarkable capabilities in novel view rendering tasks and exhibits significant potential for multi-view optimization.However, the original 3D Gaussian Splatting lacks color representation for inputs in low-light environments. Simply using enhanced images as inputs would lead to issues with multi-view consistency, and current single-view enhancement systems rely on pre-trained data, lacking scene generalization. These problems limit the application of 3D Gaussian Splatting in low-light conditions in the field of robotics, including high-fidelity modeling and feature matching. To address these challenges, we propose an unsupervised multi-view stereoscopic system based on Gaussian Splatting, called Low-Light Gaussian Splatting (LLGS). This system aims to enhance images in low-light environments while reconstructing the scene. Our method introduces a decomposable Gaussian representation called M-Color, which separately characterizes color information for targeted enhancement. Furthermore, we propose an unsupervised optimization method with zero-knowledge priors, using direction-based enhancement to ensure multi-view consistency. Experiments conducted on real-world datasets demonstrate that our system outperforms state-of-the-art methods in both low-light enhancement and 3D Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12096v2">PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Camera Ready of CVPR2025. Project Page: https://chengzhag.github.io/publication/pansplat/ Code: https://github.com/chengzhag/PanSplat
    </div>
    <details class="paper-abstract">
      With the advent of portable 360{\deg} cameras, panorama has gained significant attention in applications like virtual reality (VR), virtual tours, robotics, and autonomous driving. As a result, wide-baseline panorama view synthesis has emerged as a vital task, where high resolution, fast inference, and memory efficiency are essential. Nevertheless, existing methods are typically constrained to lower resolutions (512 $\times$ 1024) due to demanding memory and computational requirements. In this paper, we present PanSplat, a generalizable, feed-forward approach that efficiently supports resolution up to 4K (2048 $\times$ 4096). Our approach features a tailored spherical 3D Gaussian pyramid with a Fibonacci lattice arrangement, enhancing image quality while reducing information redundancy. To accommodate the demands of high resolution, we propose a pipeline that integrates a hierarchical spherical cost volume and Gaussian heads with local operations, enabling two-step deferred backpropagation for memory-efficient training on a single A100 GPU. Experiments demonstrate that PanSplat achieves state-of-the-art results with superior efficiency and image quality across both synthetic and real-world datasets. Code is available at https://github.com/chengzhag/PanSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19654v2">TexGaussian: Generating High-quality PBR Material via Octree-based 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 CVPR 2025. Project Page: https://3d-aigc.github.io/TexGaussian
    </div>
    <details class="paper-abstract">
      Physically Based Rendering (PBR) materials play a crucial role in modern graphics, enabling photorealistic rendering across diverse environment maps. Developing an effective and efficient algorithm that is capable of automatically generating high-quality PBR materials rather than RGB texture for 3D meshes can significantly streamline the 3D content creation. Most existing methods leverage pre-trained 2D diffusion models for multi-view image synthesis, which often leads to severe inconsistency between the generated textures and input 3D meshes. This paper presents TexGaussian, a novel method that uses octant-aligned 3D Gaussian Splatting for rapid PBR material generation. Specifically, we place each 3D Gaussian on the finest leaf node of the octree built from the input 3D mesh to render the multi-view images not only for the albedo map but also for roughness and metallic. Moreover, our model is trained in a regression manner instead of diffusion denoising, capable of generating the PBR material for a 3D mesh in a single feed-forward process. Extensive experiments on publicly available benchmarks demonstrate that our method synthesizes more visually pleasing PBR materials and runs faster than previous methods in both unconditional and text-conditional scenarios, exhibiting better consistency with the given geometry. Our code and trained models are available at https://3d-aigc.github.io/TexGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18108v1">Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) autonomous driving (AD) models require diverse, high-quality data to perform well across various driving scenarios. However, collecting large-scale real-world data is expensive and time-consuming, making high-fidelity synthetic data essential for enhancing data diversity and model robustness. Existing driving simulators for synthetic data generation have significant limitations: game-engine-based simulators struggle to produce realistic sensor data, while NeRF-based and diffusion-based methods face efficiency challenges. Additionally, recent simulators designed for closed-loop evaluation provide limited interaction with other vehicles, failing to simulate complex real-world traffic dynamics. To address these issues, we introduce SceneCrafter, a realistic, interactive, and efficient AD simulator based on 3D Gaussian Splatting (3DGS). SceneCrafter not only efficiently generates realistic driving logs across diverse traffic scenarios but also enables robust closed-loop evaluation of end-to-end models. Experimental results demonstrate that SceneCrafter serves as both a reliable evaluation platform and a efficient data generator that significantly improves end-to-end model generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18107v1">PanoGS: Gaussian-based Panoptic Segmentation for 3D Open Vocabulary Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has shown encouraging performance for open vocabulary scene understanding tasks. However, previous methods cannot distinguish 3D instance-level information, which usually predicts a heatmap between the scene feature and text query. In this paper, we propose PanoGS, a novel and effective 3D panoptic open vocabulary scene understanding approach. Technically, to learn accurate 3D language features that can scale to large indoor scenarios, we adopt the pyramid tri-plane to model the latent continuous parametric feature space and use a 3D feature decoder to regress the multi-view fused 2D feature cloud. Besides, we propose language-guided graph cuts that synergistically leverage reconstructed geometry and learned language cues to group 3D Gaussian primitives into a set of super-primitives. To obtain 3D consistent instance, we perform graph clustering based segmentation with SAM-guided edge affinity computation between different super-primitives. Extensive experiments on widely used datasets show better or more competitive performance on 3D panoptic open vocabulary scene understanding. Project page: \href{https://zju3dv.github.io/panogs}{https://zju3dv.github.io/panogs}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11752v2">Deformable Radial Kernel Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has emerged as a robust technique for representing 3D scenes, enabling real-time rasterization and high-fidelity rendering. However, Gaussians' inherent radial symmetry and smoothness constraints limit their ability to represent complex shapes, often requiring thousands of primitives to approximate detailed geometry. We introduce Deformable Radial Kernel (DRK), which extends Gaussian splatting into a more general and flexible framework. Through learnable radial bases with adjustable angles and scales, DRK efficiently models diverse shape primitives while enabling precise control over edge sharpness and boundary curvature. iven DRK's planar nature, we further develop accurate ray-primitive intersection computation for depth sorting and introduce efficient kernel culling strategies for improved rasterization efficiency. Extensive experiments demonstrate that DRK outperforms existing methods in both representation efficiency and rendering quality, achieving state-of-the-art performance while dramatically reducing primitive count.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18073v1">PanopticSplatting: End-to-End Panoptic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Open-vocabulary panoptic reconstruction is a challenging task for simultaneous scene reconstruction and understanding. Recently, methods have been proposed for 3D scene understanding based on Gaussian splatting. However, these methods are multi-staged, suffering from the accumulated errors and the dependence of hand-designed components. To streamline the pipeline and achieve global optimization, we propose PanopticSplatting, an end-to-end system for open-vocabulary panoptic reconstruction. Our method introduces query-guided Gaussian segmentation with local cross attention, lifting 2D instance masks without cross-frame association in an end-to-end way. The local cross attention within view frustum effectively reduces the training memory, making our model more accessible to large scenes with more Gaussians and objects. In addition, to address the challenge of noisy labels in 2D pseudo masks, we propose label blending to promote consistent 3D segmentation with less noisy floaters, as well as label warping on 2D predictions which enhances multi-view coherence and segmentation accuracy. Our method demonstrates strong performances in 3D scene panoptic reconstruction on the ScanNet-V2 and ScanNet++ datasets, compared with both NeRF-based and Gaussian-based panoptic reconstruction methods. Moreover, PanopticSplatting can be easily generalized to numerous variants of Gaussian splatting, and we demonstrate its robustness on different Gaussian base models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15392v3">EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Project Page: https://lbh666.github.io/ef-3dgs/
    </div>
    <details class="paper-abstract">
      Scene reconstruction from casually captured videos has wide applications in real-world scenarios. With recent advancements in differentiable rendering techniques, several methods have attempted to simultaneously optimize scene representations (NeRF or 3DGS) and camera poses. Despite recent progress, existing methods relying on traditional camera input tend to fail in high-speed (or equivalently low-frame-rate) scenarios. Event cameras, inspired by biological vision, record pixel-wise intensity changes asynchronously with high temporal resolution, providing valuable scene and motion information in blind inter-frame intervals. In this paper, we introduce the event camera to aid scene construction from a casually captured video for the first time, and propose Event-Aided Free-Trajectory 3DGS, called EF-3DGS, which seamlessly integrates the advantages of event cameras into 3DGS through three key components. First, we leverage the Event Generation Model (EGM) to fuse events and frames, supervising the rendered views observed by the event stream. Second, we adopt the Contrast Maximization (CMax) framework in a piece-wise manner to extract motion information by maximizing the contrast of the Image of Warped Events (IWE), thereby calibrating the estimated poses. Besides, based on the Linear Event Generation Model (LEGM), the brightness information encoded in the IWE is also utilized to constrain the 3DGS in the gradient domain. Third, to mitigate the absence of color information of events, we introduce photometric bundle adjustment (PBA) to ensure view consistency across events and frames. We evaluate our method on the public Tanks and Temples benchmark and a newly collected real-world dataset, RealEv-DAVIS. Our project page is https://lbh666.github.io/ef-3dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18052v1">SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Our code, model, and dataset will be released at https://github.com/unique1i/SceneSplat
    </div>
    <details class="paper-abstract">
      Recognizing arbitrary or previously unseen categories is essential for comprehensive real-world 3D scene understanding. Currently, all existing methods rely on 2D or textual modalities during training, or together at inference. This highlights a clear absence of a model capable of processing 3D data alone for learning semantics end-to-end, along with the necessary data to train such a model. Meanwhile, 3D Gaussian Splatting (3DGS) has emerged as the de facto standard for 3D scene representation across various vision tasks. However, effectively integrating semantic reasoning into 3DGS in a generalizable fashion remains an open challenge. To address these limitations we introduce SceneSplat, to our knowledge the first large-scale 3D indoor scene understanding approach that operates natively on 3DGS. Furthermore, we propose a self-supervised learning scheme that unlocks rich 3D feature learning from unlabeled scenes. In order to power the proposed methods, we introduce SceneSplat-7K, the first large-scale 3DGS dataset for indoor scenes, comprising of 6868 scenes derived from 7 established datasets like ScanNet, Matterport3D, etc. Generating SceneSplat-7K required computational resources equivalent to 119 GPU-days on an L4 GPU, enabling standardized benchmarking for 3DGS-based reasoning for indoor scenes. Our exhaustive experiments on SceneSplat-7K demonstrate the significant benefit of the proposed methods over the established baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17973v1">PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Project Page: https://jianghanxiao.github.io/phystwin-web/
    </div>
    <details class="paper-abstract">
      Creating a physical digital twin of a real-world object has immense potential in robotics, content creation, and XR. In this paper, we present PhysTwin, a novel framework that uses sparse videos of dynamic objects under interaction to produce a photo- and physically realistic, real-time interactive virtual replica. Our approach centers on two key components: (1) a physics-informed representation that combines spring-mass models for realistic physical simulation, generative shape models for geometry, and Gaussian splats for rendering; and (2) a novel multi-stage, optimization-based inverse modeling framework that reconstructs complete geometry, infers dense physical properties, and replicates realistic appearance from videos. Our method integrates an inverse physics framework with visual perception cues, enabling high-fidelity reconstruction even from partial, occluded, and limited viewpoints. PhysTwin supports modeling various deformable objects, including ropes, stuffed animals, cloth, and delivery packages. Experiments show that PhysTwin outperforms competing methods in reconstruction, rendering, future prediction, and simulation under novel interactions. We further demonstrate its applications in interactive real-time simulation and model-based robotic motion planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14277v2">Dense-SfM: Structure from Motion with Dense Consistent Matching</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      We present Dense-SfM, a novel Structure from Motion (SfM) framework designed for dense and accurate 3D reconstruction from multi-view images. Sparse keypoint matching, which traditional SfM methods often rely on, limits both accuracy and point density, especially in texture-less areas. Dense-SfM addresses this limitation by integrating dense matching with a Gaussian Splatting (GS) based track extension which gives more consistent, longer feature tracks. To further improve reconstruction accuracy, Dense-SfM is equipped with a multi-view kernelized matching module leveraging transformer and Gaussian Process architectures, for robust track refinement across multi-views. Evaluations on the ETH3D and Texture-Poor SfM datasets show that Dense-SfM offers significant improvements in accuracy and density over state-of-the-art methods. Project page: https://icetea-cv.github.io/densesfm/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18073v1">PanopticSplatting: End-to-End Panoptic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Open-vocabulary panoptic reconstruction is a challenging task for simultaneous scene reconstruction and understanding. Recently, methods have been proposed for 3D scene understanding based on Gaussian splatting. However, these methods are multi-staged, suffering from the accumulated errors and the dependence of hand-designed components. To streamline the pipeline and achieve global optimization, we propose PanopticSplatting, an end-to-end system for open-vocabulary panoptic reconstruction. Our method introduces query-guided Gaussian segmentation with local cross attention, lifting 2D instance masks without cross-frame association in an end-to-end way. The local cross attention within view frustum effectively reduces the training memory, making our model more accessible to large scenes with more Gaussians and objects. In addition, to address the challenge of noisy labels in 2D pseudo masks, we propose label blending to promote consistent 3D segmentation with less noisy floaters, as well as label warping on 2D predictions which enhances multi-view coherence and segmentation accuracy. Our method demonstrates strong performances in 3D scene panoptic reconstruction on the ScanNet-V2 and ScanNet++ datasets, compared with both NeRF-based and Gaussian-based panoptic reconstruction methods. Moreover, PanopticSplatting can be easily generalized to numerous variants of Gaussian splatting, and we demonstrate its robustness on different Gaussian base models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17798v1">GaussianFocus: Constrained Attention Focus for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recent developments in 3D reconstruction and neural rendering have significantly propelled the capabilities of photo-realistic 3D scene rendering across various academic and industrial fields. The 3D Gaussian Splatting technique, alongside its derivatives, integrates the advantages of primitive-based and volumetric representations to deliver top-tier rendering quality and efficiency. Despite these advancements, the method tends to generate excessive redundant noisy Gaussians overfitted to every training view, which degrades the rendering quality. Additionally, while 3D Gaussian Splatting excels in small-scale and object-centric scenes, its application to larger scenes is hindered by constraints such as limited video memory, excessive optimization duration, and variable appearance across views. To address these challenges, we introduce GaussianFocus, an innovative approach that incorporates a patch attention algorithm to refine rendering quality and implements a Gaussian constraints strategy to minimize redundancy. Moreover, we propose a subdivision reconstruction strategy for large-scale scenes, dividing them into smaller, manageable blocks for individual training. Our results indicate that GaussianFocus significantly reduces unnecessary Gaussians and enhances rendering quality, surpassing existing State-of-The-Art (SoTA) methods. Furthermore, we demonstrate the capability of our approach to effectively manage and render large scenes, such as urban environments, whilst maintaining high fidelity in the visual output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13047v2">Gaussian Splatting for Efficient Satellite Image Photogrammetry</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has emerged as a strong alternative to NeRF, demonstrating impressive 3D modeling capabilities while requiring only a fraction of the training and rendering time. In this paper, we show how the standard Gaussian splatting framework can be adapted for remote sensing, retaining its high efficiency. This enables us to achieve state-of-the-art performance in just a few minutes, compared to the day-long optimization required by the best-performing NeRF-based Earth observation methods. The proposed framework incorporates remote-sensing improvements from EO-NeRF, such as radiometric correction and shadow modeling, while introducing novel components, including sparsity, view consistency, and opacity regularizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01949v2">LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recently, the field of text-guided 3D scene generation has garnered significant attention. High-quality generation that aligns with physical realism and high controllability is crucial for practical 3D scene applications. However, existing methods face fundamental limitations: (i) difficulty capturing complex relationships between multiple objects described in the text, (ii) inability to generate physically plausible scene layouts, and (iii) lack of controllability and extensibility in compositional scenes. In this paper, we introduce LayoutDreamer, a framework that leverages 3D Gaussian Splatting (3DGS) to facilitate high-quality, physically consistent compositional scene generation guided by text. Specifically, given a text prompt, we convert it into a directed scene graph and adaptively adjust the density and layout of the initial compositional 3D Gaussians. Subsequently, dynamic camera adjustments are made based on the training focal point to ensure entity-level generation quality. Finally, by extracting directed dependencies from the scene graph, we tailor physical and layout energy to ensure both realism and flexibility. Comprehensive experiments demonstrate that LayoutDreamer outperforms other compositional scene generation quality and semantic alignment methods. Specifically, it achieves state-of-the-art (SOTA) performance in the multiple objects generation metric of T3Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17733v1">GS-LTS: 3D Gaussian Splatting-Based Adaptive Modeling for Long-Term Service Robots</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has garnered significant attention in robotics for its explicit, high fidelity dense scene representation, demonstrating strong potential for robotic applications. However, 3DGS-based methods in robotics primarily focus on static scenes, with limited attention to the dynamic scene changes essential for long-term service robots. These robots demand sustained task execution and efficient scene updates-challenges current approaches fail to meet. To address these limitations, we propose GS-LTS (Gaussian Splatting for Long-Term Service), a 3DGS-based system enabling indoor robots to manage diverse tasks in dynamic environments over time. GS-LTS detects scene changes (e.g., object addition or removal) via single-image change detection, employs a rule-based policy to autonomously collect multi-view observations, and efficiently updates the scene representation through Gaussian editing. Additionally, we propose a simulation-based benchmark that automatically generates scene change data as compact configuration scripts, providing a standardized, user-friendly evaluation benchmark. Experimental results demonstrate GS-LTS's advantages in reconstruction, navigation, and superior scene updates-faster and higher quality than the image training baseline-advancing 3DGS for long-term robotic operations. Code and benchmark are available at: https://vipl-vsu.github.io/3DGS-LTS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06235v2">StreamGS: Online Generalizable Gaussian Splatting Reconstruction for Unposed Image Streams</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The advent of 3D Gaussian Splatting (3DGS) has advanced 3D scene reconstruction and novel view synthesis. With the growing interest of interactive applications that need immediate feedback, online 3DGS reconstruction in real-time is in high demand. However, none of existing methods yet meet the demand due to three main challenges: the absence of predetermined camera parameters, the need for generalizable 3DGS optimization, and the necessity of reducing redundancy. We propose StreamGS, an online generalizable 3DGS reconstruction method for unposed image streams, which progressively transform image streams to 3D Gaussian streams by predicting and aggregating per-frame Gaussians. Our method overcomes the limitation of the initial point reconstruction \cite{dust3r} in tackling out-of-domain (OOD) issues by introducing a content adaptive refinement. The refinement enhances cross-frame consistency by establishing reliable pixel correspondences between adjacent frames. Such correspondences further aid in merging redundant Gaussians through cross-frame feature aggregation. The density of Gaussians is thereby reduced, empowering online reconstruction by significantly lowering computational and memory costs. Extensive experiments on diverse datasets have demonstrated that StreamGS achieves quality on par with optimization-based approaches but does so 150 times faster, and exhibits superior generalizability in handling OOD scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12552v3">MTGS: Multi-Traversal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Multi-traversal data, commonly collected through daily commutes or by self-driving fleets, provides multiple viewpoints for scene reconstruction within a road block. This data offers significant potential for high-quality novel view synthesis, which is crucial for applications such as autonomous vehicle simulators. However, inherent challenges in multi-traversal data often result in suboptimal reconstruction quality, including variations in appearance and the presence of dynamic objects. To address these issues, we propose Multi-Traversal Gaussian Splatting (MTGS), a novel approach that reconstructs high-quality driving scenes from arbitrarily collected multi-traversal data by modeling a shared static geometry while separately handling dynamic elements and appearance variations. Our method employs a multi-traversal dynamic scene graph with a shared static node and traversal-specific dynamic nodes, complemented by color correction nodes with learnable spherical harmonics coefficient residuals. This approach enables high-fidelity novel view synthesis and provides flexibility to navigate any viewpoint. We conduct extensive experiments on a large-scale driving dataset, nuPlan, with multi-traversal data. Our results demonstrate that MTGS improves LPIPS by 23.5% and geometry accuracy by 46.3% compared to single-traversal baselines. The code and data would be available to the public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.13047v2">Gaussian Splatting for Efficient Satellite Image Photogrammetry</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recently, Gaussian splatting has emerged as a strong alternative to NeRF, demonstrating impressive 3D modeling capabilities while requiring only a fraction of the training and rendering time. In this paper, we show how the standard Gaussian splatting framework can be adapted for remote sensing, retaining its high efficiency. This enables us to achieve state-of-the-art performance in just a few minutes, compared to the day-long optimization required by the best-performing NeRF-based Earth observation methods. The proposed framework incorporates remote-sensing improvements from EO-NeRF, such as radiometric correction and shadow modeling, while introducing novel components, including sparsity, view consistency, and opacity regularizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16346v2">SOUS VIDE: Cooking Visual Drone Navigation Policies in a Gaussian Splatting Vacuum</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      We propose a new simulator, training approach, and policy architecture, collectively called SOUS VIDE, for end-to-end visual drone navigation. Our trained policies exhibit zero-shot sim-to-real transfer with robust real-world performance using only onboard perception and computation. Our simulator, called FiGS, couples a computationally simple drone dynamics model with a high visual fidelity Gaussian Splatting scene reconstruction. FiGS can quickly simulate drone flights producing photorealistic images at up to 130 fps. We use FiGS to collect 100k-300k image/state-action pairs from an expert MPC with privileged state and dynamics information, randomized over dynamics parameters and spatial disturbances. We then distill this expert MPC into an end-to-end visuomotor policy with a lightweight neural architecture, called SV-Net. SV-Net processes color image, optical flow and IMU data streams into low-level thrust and body rate commands at 20 Hz onboard a drone. Crucially, SV-Net includes a learned module for low-level control that adapts at runtime to variations in drone dynamics. In a campaign of 105 hardware experiments, we show SOUS VIDE policies to be robust to 30% mass variations, 40 m/s wind gusts, 60% changes in ambient brightness, shifting or removing objects from the scene, and people moving aggressively through the drone's visual field. Code, data, and experiment videos can be found on our project page: https://stanfordmsl.github.io/SousVide/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12231v2">PUGS: Zero-shot Physical Understanding with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 ICRA 2025, Project page: https://evernorif.github.io/PUGS/
    </div>
    <details class="paper-abstract">
      Current robotic systems can understand the categories and poses of objects well. But understanding physical properties like mass, friction, and hardness, in the wild, remains challenging. We propose a new method that reconstructs 3D objects using the Gaussian splatting representation and predicts various physical properties in a zero-shot manner. We propose two techniques during the reconstruction phase: a geometry-aware regularization loss function to improve the shape quality and a region-aware feature contrastive loss function to promote region affinity. Two other new techniques are designed during inference: a feature-based property propagation module and a volume integration module tailored for the Gaussian representation. Our framework is named as zero-shot physical understanding with Gaussian splatting, or PUGS. PUGS achieves new state-of-the-art results on the standard benchmark of ABO-500 mass prediction. We provide extensive quantitative ablations and qualitative visualization to demonstrate the mechanism of our designs. We show the proposed methodology can help address challenging real-world grasping tasks. Our codes, data, and models are available at https://github.com/EverNorif/PUGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17032v1">TaoAvatar: Real-Time Lifelike Full-Body Talking Avatars for Augmented Reality via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 Accepted by CVPR 2025, project page: https://PixelAI-Team.github.io/TaoAvatar
    </div>
    <details class="paper-abstract">
      Realistic 3D full-body talking avatars hold great potential in AR, with applications ranging from e-commerce live streaming to holographic communication. Despite advances in 3D Gaussian Splatting (3DGS) for lifelike avatar creation, existing methods struggle with fine-grained control of facial expressions and body movements in full-body talking tasks. Additionally, they often lack sufficient details and cannot run in real-time on mobile devices. We present TaoAvatar, a high-fidelity, lightweight, 3DGS-based full-body talking avatar driven by various signals. Our approach starts by creating a personalized clothed human parametric template that binds Gaussians to represent appearances. We then pre-train a StyleUnet-based network to handle complex pose-dependent non-rigid deformation, which can capture high-frequency appearance details but is too resource-intensive for mobile devices. To overcome this, we "bake" the non-rigid deformations into a lightweight MLP-based network using a distillation technique and develop blend shapes to compensate for details. Extensive experiments show that TaoAvatar achieves state-of-the-art rendering quality while running in real-time across various devices, maintaining 90 FPS on high-definition stereo devices such as the Apple Vision Pro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03473v2">UrbanGS: Semantic-Guided Gaussian Splatting for Urban Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Reconstructing urban scenes is challenging due to their complex geometries and the presence of potentially dynamic objects. 3D Gaussian Splatting (3DGS)-based methods have shown strong performance, but existing approaches often incorporate manual 3D annotations to improve dynamic object modeling, which is impractical due to high labeling costs. Some methods leverage 4D Gaussian Splatting (4DGS) to represent the entire scene, but they treat static and dynamic objects uniformly, leading to unnecessary updates for static elements and ultimately degrading reconstruction quality. To address these issues, we propose UrbanGS, which leverages 2D semantic maps and an existing dynamic Gaussian approach to distinguish static objects from the scene, enabling separate processing of definite static and potentially dynamic elements. Specifically, for definite static regions, we enforce global consistency to prevent unintended changes in dynamic Gaussian and introduce a K-nearest neighbor (KNN)-based regularization to improve local coherence on low-textured ground surfaces. Notably, for potentially dynamic objects, we aggregate temporal information using learnable time embeddings, allowing each Gaussian to model deformations over time. Extensive experiments on real-world datasets demonstrate that our approach outperforms state-of-the-art methods in reconstruction quality and efficiency, accurately preserving static content while capturing dynamic elements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16979v1">Instant Gaussian Stream: Fast and Generalizable Streaming of Dynamic Scene Reconstruction via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Building Free-Viewpoint Videos in a streaming manner offers the advantage of rapid responsiveness compared to offline training methods, greatly enhancing user experience. However, current streaming approaches face challenges of high per-frame reconstruction time (10s+) and error accumulation, limiting their broader application. In this paper, we propose Instant Gaussian Stream (IGS), a fast and generalizable streaming framework, to address these issues. First, we introduce a generalized Anchor-driven Gaussian Motion Network, which projects multi-view 2D motion features into 3D space, using anchor points to drive the motion of all Gaussians. This generalized Network generates the motion of Gaussians for each target frame in the time required for a single inference. Second, we propose a Key-frame-guided Streaming Strategy that refines each key frame, enabling accurate reconstruction of temporally complex scenes while mitigating error accumulation. We conducted extensive in-domain and cross-domain evaluations, demonstrating that our approach can achieve streaming with a average per-frame reconstruction time of 2s+, alongside a enhancement in view synthesis quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16964v1">DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Drones have become essential tools for reconstructing wild scenes due to their outstanding maneuverability. Recent advances in radiance field methods have achieved remarkable rendering quality, providing a new avenue for 3D reconstruction from drone imagery. However, dynamic distractors in wild environments challenge the static scene assumption in radiance fields, while limited view constraints hinder the accurate capture of underlying scene geometry. To address these challenges, we introduce DroneSplat, a novel framework designed for robust 3D reconstruction from in-the-wild drone imagery. Our method adaptively adjusts masking thresholds by integrating local-global segmentation heuristics with statistical approaches, enabling precise identification and elimination of dynamic distractors in static scenes. We enhance 3D Gaussian Splatting with multi-view stereo predictions and a voxel-guided optimization strategy, supporting high-quality rendering under limited view constraints. For comprehensive evaluation, we provide a drone-captured 3D reconstruction dataset encompassing both dynamic and static scenes. Extensive experiments demonstrate that DroneSplat outperforms both 3DGS and NeRF baselines in handling in-the-wild drone imagery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16924v1">Optimized Minimal 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 Project page: https://maincold2.github.io/omg/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation for real-time, high-performance rendering, enabling a wide range of applications. However, representing 3D scenes with numerous explicit Gaussian primitives imposes significant storage and memory overhead. Recent studies have shown that high-quality rendering can be achieved with a substantially reduced number of Gaussians when represented with high-precision attributes. Nevertheless, existing 3DGS compression methods still rely on a relatively large number of Gaussians, focusing primarily on attribute compression. This is because a smaller set of Gaussians becomes increasingly sensitive to lossy attribute compression, leading to severe quality degradation. Since the number of Gaussians is directly tied to computational costs, it is essential to reduce the number of Gaussians effectively rather than only optimizing storage. In this paper, we propose Optimized Minimal Gaussians representation (OMG), which significantly reduces storage while using a minimal number of primitives. First, we determine the distinct Gaussian from the near ones, minimizing redundancy without sacrificing quality. Second, we propose a compact and precise attribute representation that efficiently captures both continuity and irregularity among primitives. Additionally, we propose a sub-vector quantization technique for improved irregularity representation, maintaining fast training with a negligible codebook size. Extensive experiments demonstrate that OMG reduces storage requirements by nearly 50% compared to the previous state-of-the-art and enables 600+ FPS rendering while maintaining high rendering quality. Our source code is available at https://maincold2.github.io/omg/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00239v2">Aquatic-GS: A Hybrid 3D Representation for Underwater Scenes</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 13 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Representing underwater 3D scenes is a valuable yet complex task, as attenuation and scattering effects during underwater imaging significantly couple the information of the objects and the water. This coupling presents a significant challenge for existing methods in effectively representing both the objects and the water medium simultaneously. To address this challenge, we propose Aquatic-GS, a hybrid 3D representation approach for underwater scenes that effectively represents both the objects and the water medium. Specifically, we construct a Neural Water Field (NWF) to implicitly model the water parameters, while extending the latest 3D Gaussian Splatting (3DGS) to model the objects explicitly. Both components are integrated through a physics-based underwater image formation model to represent complex underwater scenes. Moreover, to construct more precise scene geometry and details, we design a Depth-Guided Optimization (DGO) mechanism that uses a pseudo-depth map as auxiliary guidance. After optimization, Aquatic-GS enables the rendering of novel underwater viewpoints and supports restoring the true appearance of underwater scenes, as if the water medium were absent. Extensive experiments on both simulated and real-world datasets demonstrate that Aquatic-GS surpasses state-of-the-art underwater 3D representation methods, achieving better rendering quality and real-time rendering performance with a 410x increase in speed. Furthermore, regarding underwater image restoration, Aquatic-GS outperforms representative dewatering methods in color correction, detail recovery, and stability. Our models, code, and datasets can be accessed at https://aquaticgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04832v3">Neural Representation for Wireless Radiation Field Reconstruction: A 3D Gaussian Splatting Approach</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 This work has been submitted to the IEEE journals for possible publication. The code is available at https://github.com/wenchaozheng/WRF-GSplus
    </div>
    <details class="paper-abstract">
      Wireless channel modeling plays a pivotal role in designing, analyzing, and optimizing wireless communication systems. Nevertheless, developing an effective channel modeling approach has been a long-standing challenge. This issue has been escalated due to denser network deployment, larger antenna arrays, and broader bandwidth in next-generation networks. To address this challenge, we put forth WRF-GS, a novel framework for channel modeling based on wireless radiation field (WRF) reconstruction using 3D Gaussian splatting (3D-GS). WRF-GS employs 3D Gaussian primitives and neural networks to capture the interactions between the environment and radio signals, enabling efficient WRF reconstruction and visualization of the propagation characteristics. The reconstructed WRF can then be used to synthesize the spatial spectrum for comprehensive wireless channel characterization. While WRF-GS demonstrates remarkable effectiveness, it faces limitations in capturing high-frequency signal variations caused by complex multipath effects. To overcome these limitations, we propose WRF-GS+, an enhanced framework that integrates electromagnetic wave physics into the neural network design. WRF-GS+ leverages deformable 3D Gaussians to model both static and dynamic components of the WRF, significantly improving its ability to characterize signal variations. In addition, WRF-GS+ enhances the splatting process by simplifying the 3D-GS modeling process and improving computational efficiency. Experimental results demonstrate that both WRF-GS and WRF-GS+ outperform baselines for spatial spectrum synthesis, including ray tracing and other deep-learning approaches. Notably, WRF-GS+ achieves state-of-the-art performance in the received signal strength indication (RSSI) and channel state information (CSI) prediction tasks, surpassing existing methods by more than 0.7 dB and 3.36 dB, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10148v3">3D Student Splatting and Scooping</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) provides a new framework for novel view synthesis, and has spiked a new wave of research in neural rendering and related applications. As 3DGS is becoming a foundational component of many models, any improvement on 3DGS itself can bring huge benefits. To this end, we aim to improve the fundamental paradigm and formulation of 3DGS. We argue that as an unnormalized mixture model, it needs to be neither Gaussians nor splatting. We subsequently propose a new mixture model consisting of flexible Student's t distributions, with both positive (splatting) and negative (scooping) densities. We name our model Student Splatting and Scooping, or SSS. When providing better expressivity, SSS also poses new challenges in learning. Therefore, we also propose a new principled sampling approach for optimization. Through exhaustive evaluation and comparison, across multiple datasets, settings, and metrics, we demonstrate that SSS outperforms existing methods in terms of quality and parameter efficiency, e.g. achieving matching or better quality with similar numbers of components, and obtaining comparable results while reducing the component number by as much as 82%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17574v1">Is there anything left? Measuring semantic residuals of objects removed from 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Searching in and editing 3D scenes has become extremely intuitive with trainable scene representations that allow linking human concepts to elements in the scene. These operations are often evaluated on the basis of how accurately the searched element is segmented or extracted from the scene. In this paper, we address the inverse problem, that is, how much of the searched element remains in the scene after it is removed. This question is particularly important in the context of privacy-preserving mapping when a user reconstructs a 3D scene and wants to remove private elements before sharing the map. To the best of our knowledge, this is the first work to address this question. To answer this, we propose a quantitative evaluation that measures whether a removal operation leaves object residuals that can be reasoned over. The scene is not private when such residuals are present. Experiments on state-of-the-art scene representations show that the proposed metrics are meaningful and consistent with the user study that we also present. We also propose a method to refine the removal based on spatial and semantic consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13558v2">GoDe: Gaussians on Demand for Progressive Level of Detail and Scalable Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting enhances real-time performance in novel view synthesis by representing scenes with mixtures of Gaussians and utilizing differentiable rasterization. However, it typically requires large storage capacity and high VRAM, demanding the design of effective pruning and compression techniques. Existing methods, while effective in some scenarios, struggle with scalability and fail to adapt models based on critical factors such as computing capabilities or bandwidth, requiring to re-train the model under different configurations. In this work, we propose a novel, model-agnostic technique that organizes Gaussians into several hierarchical layers, enabling progressive Level of Detail (LoD) strategy. This method, combined with recent approach of compression of 3DGS, allows a single model to instantly scale across several compression ratios, with minimal to none impact to quality compared to a single non-scalable model and without requiring re-training. We validate our approach on typical datasets and benchmarks, showcasing low distortion and substantial gains in terms of scalability and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17491v1">Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 submitted to ICCV 2025
    </div>
    <details class="paper-abstract">
      LiDARs provide accurate geometric measurements, making them valuable for ego-motion estimation and reconstruction tasks. Although its success, managing an accurate and lightweight representation of the environment still poses challenges. Both classic and NeRF-based solutions have to trade off accuracy over memory and processing times. In this work, we build on recent advancements in Gaussian Splatting methods to develop a novel LiDAR odometry and mapping pipeline that exclusively relies on Gaussian primitives for its scene representation. Leveraging spherical projection, we drive the refinement of the primitives uniquely from LiDAR measurements. Experiments show that our approach matches the current registration performance, while achieving SOTA results for mapping tasks with minimal GPU requirements. This efficiency makes it a strong candidate for further exploration and potential adoption in real-time robotics estimation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17486v1">ProtoGS: Efficient and High-Quality Rendering with 3D Gaussian Prototypes</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in novel view synthesis but is limited by the substantial number of Gaussian primitives required, posing challenges for deployment on lightweight devices. Recent methods address this issue by compressing the storage size of densified Gaussians, yet fail to preserve rendering quality and efficiency. To overcome these limitations, we propose ProtoGS to learn Gaussian prototypes to represent Gaussian primitives, significantly reducing the total Gaussian amount without sacrificing visual quality. Our method directly uses Gaussian prototypes to enable efficient rendering and leverage the resulting reconstruction loss to guide prototype learning. To further optimize memory efficiency during training, we incorporate structure-from-motion (SfM) points as anchor points to group Gaussian primitives. Gaussian prototypes are derived within each group by clustering of K-means, and both the anchor points and the prototypes are optimized jointly. Our experiments on real-world and synthetic datasets prove that we outperform existing methods, achieving a substantial reduction in the number of Gaussians, and enabling high rendering speed while maintaining or even enhancing rendering fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17491v1">Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 submitted to ICCV 2025
    </div>
    <details class="paper-abstract">
      LiDARs provide accurate geometric measurements, making them valuable for ego-motion estimation and reconstruction tasks. Although its success, managing an accurate and lightweight representation of the environment still poses challenges. Both classic and NeRF-based solutions have to trade off accuracy over memory and processing times. In this work, we build on recent advancements in Gaussian Splatting methods to develop a novel LiDAR odometry and mapping pipeline that exclusively relies on Gaussian primitives for its scene representation. Leveraging spherical projection, we drive the refinement of the primitives uniquely from LiDAR measurements. Experiments show that our approach matches the current registration performance, while achieving SOTA results for mapping tasks with minimal GPU requirements. This efficiency makes it a strong candidate for further exploration and potential adoption in real-time robotics estimation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12231v2">PUGS: Zero-shot Physical Understanding with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 ICRA 2025, Project page: https://evernorif.github.io/PUGS/
    </div>
    <details class="paper-abstract">
      Current robotic systems can understand the categories and poses of objects well. But understanding physical properties like mass, friction, and hardness, in the wild, remains challenging. We propose a new method that reconstructs 3D objects using the Gaussian splatting representation and predicts various physical properties in a zero-shot manner. We propose two techniques during the reconstruction phase: a geometry-aware regularization loss function to improve the shape quality and a region-aware feature contrastive loss function to promote region affinity. Two other new techniques are designed during inference: a feature-based property propagation module and a volume integration module tailored for the Gaussian representation. Our framework is named as zero-shot physical understanding with Gaussian splatting, or PUGS. PUGS achieves new state-of-the-art results on the standard benchmark of ABO-500 mass prediction. We provide extensive quantitative ablations and qualitative visualization to demonstrate the mechanism of our designs. We show the proposed methodology can help address challenging real-world grasping tasks. Our codes, data, and models are available at https://github.com/EverNorif/PUGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16422v1">1000+ FPS 4D Gaussian Splatting for Dynamic Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) has recently gained considerable attention as a method for reconstructing dynamic scenes. Despite achieving superior quality, 4DGS typically requires substantial storage and suffers from slow rendering speed. In this work, we delve into these issues and identify two key sources of temporal redundancy. (Q1) \textbf{Short-Lifespan Gaussians}: 4DGS uses a large portion of Gaussians with short temporal span to represent scene dynamics, leading to an excessive number of Gaussians. (Q2) \textbf{Inactive Gaussians}: When rendering, only a small subset of Gaussians contributes to each frame. Despite this, all Gaussians are processed during rasterization, resulting in redundant computation overhead. To address these redundancies, we present \textbf{4DGS-1K}, which runs at over 1000 FPS on modern GPUs. For Q1, we introduce the Spatial-Temporal Variation Score, a new pruning criterion that effectively removes short-lifespan Gaussians while encouraging 4DGS to capture scene dynamics using Gaussians with longer temporal spans. For Q2, we store a mask for active Gaussians across consecutive frames, significantly reducing redundant computations in rendering. Compared to vanilla 4DGS, our method achieves a $41\times$ reduction in storage and $9\times$ faster rasterization speed on complex dynamic scenes, while maintaining comparable visual quality. Please see our project page at https://4DGS-1K.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16413v1">M3: 3D-Spatial MultiModal Memory</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 ICLR2025 homepage: https://m3-spatial-memory.github.io code: https://github.com/MaureenZOU/m3-spatial
    </div>
    <details class="paper-abstract">
      We present 3D Spatial MultiModal Memory (M3), a multimodal memory system designed to retain information about medium-sized static scenes through video sources for visual perception. By integrating 3D Gaussian Splatting techniques with foundation models, M3 builds a multimodal memory capable of rendering feature representations across granularities, encompassing a wide range of knowledge. In our exploration, we identify two key challenges in previous works on feature splatting: (1) computational constraints in storing high-dimensional features for each Gaussian primitive, and (2) misalignment or information loss between distilled features and foundation model features. To address these challenges, we propose M3 with key components of principal scene components and Gaussian memory attention, enabling efficient training and inference. To validate M3, we conduct comprehensive quantitative evaluations of feature similarity and downstream tasks, as well as qualitative visualizations to highlight the pixel trace of Gaussian memory attention. Our approach encompasses a diverse range of foundation models, including vision-language models (VLMs), perception models, and large multimodal and language models (LMMs/LLMs). Furthermore, to demonstrate real-world applicability, we deploy M3's feature field in indoor scenes on a quadruped robot. Notably, we claim that M3 is the first work to address the core compression challenges in 3D feature distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16338v1">Gaussian Graph Network: Learning Efficient and Generalizable Gaussian Representations from Multi-view Images</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis performance. While conventional methods require per-scene optimization, more recently several feed-forward methods have been proposed to generate pixel-aligned Gaussian representations with a learnable network, which are generalizable to different scenes. However, these methods simply combine pixel-aligned Gaussians from multiple views as scene representations, thereby leading to artifacts and extra memory cost without fully capturing the relations of Gaussians from different images. In this paper, we propose Gaussian Graph Network (GGN) to generate efficient and generalizable Gaussian representations. Specifically, we construct Gaussian Graphs to model the relations of Gaussian groups from different views. To support message passing at Gaussian level, we reformulate the basic graph operations over Gaussian representations, enabling each Gaussian to benefit from its connected Gaussian groups with Gaussian feature fusion. Furthermore, we design a Gaussian pooling layer to aggregate various Gaussian groups for efficient representations. We conduct experiments on the large-scale RealEstate10K and ACID datasets to demonstrate the efficiency and generalization of our method. Compared to the state-of-the-art methods, our model uses fewer Gaussians and achieves better image quality with higher rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16177v1">OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Project website: https://occlugaussian.github.io
    </div>
    <details class="paper-abstract">
      In large-scale scene reconstruction using 3D Gaussian splatting, it is common to partition the scene into multiple smaller regions and reconstruct them individually. However, existing division methods are occlusion-agnostic, meaning that each region may contain areas with severe occlusions. As a result, the cameras within those regions are less correlated, leading to a low average contribution to the overall reconstruction. In this paper, we propose an occlusion-aware scene division strategy that clusters training cameras based on their positions and co-visibilities to acquire multiple regions. Cameras in such regions exhibit stronger correlations and a higher average contribution, facilitating high-quality scene reconstruction. We further propose a region-based rendering technique to accelerate large scene rendering, which culls Gaussians invisible to the region where the viewpoint is located. Such a technique significantly speeds up the rendering without compromising quality. Extensive experiments on multiple large scenes show that our method achieves superior reconstruction results with faster rendering speed compared to existing state-of-the-art approaches. Project page: https://occlugaussian.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16502v3">GSplatLoc: Grounding Keypoint Descriptors into 3D Gaussian Splatting for Improved Visual Localization</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Project website at https://gsplatloc.github.io/
    </div>
    <details class="paper-abstract">
      Although various visual localization approaches exist, such as scene coordinate regression and camera pose regression, these methods often struggle with optimization complexity or limited accuracy. To address these challenges, we explore the use of novel view synthesis techniques, particularly 3D Gaussian Splatting (3DGS), which enables the compact encoding of both 3D geometry and scene appearance. We propose a two-stage procedure that integrates dense and robust keypoint descriptors from the lightweight XFeat feature extractor into 3DGS, enhancing performance in both indoor and outdoor environments. The coarse pose estimates are directly obtained via 2D-3D correspondences between the 3DGS representation and query image descriptors. In the second stage, the initial pose estimate is refined by minimizing the rendering-based photometric warp loss. Benchmarking on widely used indoor and outdoor datasets demonstrates improvements over recent neural rendering-based localization methods, such as NeRFMatch and PNeRFLoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06903v2">Synthetic Prior for Few-Shot Drivable Head Avatar Inversion</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted to CVPR25 Website: https://zielon.github.io/synshot/
    </div>
    <details class="paper-abstract">
      We present SynShot, a novel method for the few-shot inversion of a drivable head avatar based on a synthetic prior. We tackle three major challenges. First, training a controllable 3D generative network requires a large number of diverse sequences, for which pairs of images and high-quality tracked meshes are not always available. Second, the use of real data is strictly regulated (e.g., under the General Data Protection Regulation, which mandates frequent deletion of models and data to accommodate a situation when a participant's consent is withdrawn). Synthetic data, free from these constraints, is an appealing alternative. Third, state-of-the-art monocular avatar models struggle to generalize to new views and expressions, lacking a strong prior and often overfitting to a specific viewpoint distribution. Inspired by machine learning models trained solely on synthetic data, we propose a method that learns a prior model from a large dataset of synthetic heads with diverse identities, expressions, and viewpoints. With few input images, SynShot fine-tunes the pretrained synthetic prior to bridge the domain gap, modeling a photorealistic head avatar that generalizes to novel expressions and viewpoints. We model the head avatar using 3D Gaussian splatting and a convolutional encoder-decoder that outputs Gaussian parameters in UV texture space. To account for the different modeling complexities over parts of the head (e.g., skin vs hair), we embed the prior with explicit control for upsampling the number of per-part primitives. Compared to SOTA monocular and GAN-based methods, SynShot significantly improves novel view and expression synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04545v3">Gaussian Eigen Models for Human Heads</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted to CVPR25 Website: https://zielon.github.io/gem/
    </div>
    <details class="paper-abstract">
      Current personalized neural head avatars face a trade-off: lightweight models lack detail and realism, while high-quality, animatable avatars require significant computational resources, making them unsuitable for commodity devices. To address this gap, we introduce Gaussian Eigen Models (GEM), which provide high-quality, lightweight, and easily controllable head avatars. GEM utilizes 3D Gaussian primitives for representing the appearance combined with Gaussian splatting for rendering. Building on the success of mesh-based 3D morphable face models (3DMM), we define GEM as an ensemble of linear eigenbases for representing the head appearance of a specific subject. In particular, we construct linear bases to represent the position, scale, rotation, and opacity of the 3D Gaussians. This allows us to efficiently generate Gaussian primitives of a specific head shape by a linear combination of the basis vectors, only requiring a low-dimensional parameter vector that contains the respective coefficients. We propose to construct these linear bases (GEM) by distilling high-quality compute-intense CNN-based Gaussian avatar models that can generate expression-dependent appearance changes like wrinkles. These high-quality models are trained on multi-view videos of a subject and are distilled using a series of principal component analyses. Once we have obtained the bases that represent the animatable appearance space of a specific human, we learn a regressor that takes a single RGB image as input and predicts the low-dimensional parameter vector that corresponds to the shown facial expression. In a series of experiments, we compare GEM's self-reenactment and cross-person reenactment results to state-of-the-art 3D avatar methods, demonstrating GEM's higher visual quality and better generalization to new expressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03911v2">Multi-View Pose-Agnostic Change Localization with Zero Labels</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted at CVPR 2025
    </div>
    <details class="paper-abstract">
      Autonomous agents often require accurate methods for detecting and localizing changes in their environment, particularly when observations are captured from unconstrained and inconsistent viewpoints. We propose a novel label-free, pose-agnostic change detection method that integrates information from multiple viewpoints to construct a change-aware 3D Gaussian Splatting (3DGS) representation of the scene. With as few as 5 images of the post-change scene, our approach can learn an additional change channel in a 3DGS and produce change masks that outperform single-view techniques. Our change-aware 3D scene representation additionally enables the generation of accurate change masks for unseen viewpoints. Experimental results demonstrate state-of-the-art performance in complex multi-object scenes, achieving a 1.7x and 1.5x improvement in Mean Intersection Over Union and F1 score respectively over other baselines. We also contribute a new real-world dataset to benchmark change detection in diverse challenging scenes in the presence of lighting variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12552v2">MTGS: Multi-Traversal Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Multi-traversal data, commonly collected through daily commutes or by self-driving fleets, provides multiple viewpoints for scene reconstruction within a road block. This data offers significant potential for high-quality novel view synthesis, which is crucial for applications such as autonomous vehicle simulators. However, inherent challenges in multi-traversal data often result in suboptimal reconstruction quality, including variations in appearance and the presence of dynamic objects. To address these issues, we propose Multi-Traversal Gaussian Splatting (MTGS), a novel approach that reconstructs high-quality driving scenes from arbitrarily collected multi-traversal data by modeling a shared static geometry while separately handling dynamic elements and appearance variations. Our method employs a multi-traversal dynamic scene graph with a shared static node and traversal-specific dynamic nodes, complemented by color correction nodes with learnable spherical harmonics coefficient residuals. This approach enables high-fidelity novel view synthesis and provides flexibility to navigate any viewpoint. We conduct extensive experiments on a large-scale driving dataset, nuPlan, with multi-traversal data. Our results demonstrate that MTGS improves LPIPS by 23.5% and geometry accuracy by 46.3% compared to single-traversal baselines. The code and data would be available to the public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15908v1">Enhancing Close-up Novel View Synthesis via Pseudo-labeling</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      Recent methods, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have demonstrated remarkable capabilities in novel view synthesis. However, despite their success in producing high-quality images for viewpoints similar to those seen during training, they struggle when generating detailed images from viewpoints that significantly deviate from the training set, particularly in close-up views. The primary challenge stems from the lack of specific training data for close-up views, leading to the inability of current methods to render these views accurately. To address this issue, we introduce a novel pseudo-label-based learning strategy. This approach leverages pseudo-labels derived from existing training data to provide targeted supervision across a wide range of close-up viewpoints. Recognizing the absence of benchmarks for this specific challenge, we also present a new dataset designed to assess the effectiveness of both current and future methods in this area. Our extensive experiments demonstrate the efficacy of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20031v3">MG-SLAM: Structure Gaussian Splatting SLAM with Manhattan World Hypothesis</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Gaussian Splatting SLAMs have made significant advancements in improving the efficiency and fidelity of real-time reconstructions. However, these systems often encounter incomplete reconstructions in complex indoor environments, characterized by substantial holes due to unobserved geometry caused by obstacles or limited view angles. To address this challenge, we present Manhattan Gaussian SLAM, an RGB-D system that leverages the Manhattan World hypothesis to enhance geometric accuracy and completeness. By seamlessly integrating fused line segments derived from structured scenes, our method ensures robust tracking in textureless indoor areas. Moreover, The extracted lines and planar surface assumption allow strategic interpolation of new Gaussians in regions of missing geometry, enabling efficient scene completion. Extensive experiments conducted on both synthetic and real-world scenes demonstrate that these advancements enable our method to achieve state-of-the-art performance, marking a substantial improvement in the capabilities of Gaussian SLAM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15855v1">VideoRFSplat: Direct Scene-Level Text-to-3D Gaussian Splatting Generation with Flexible Pose and Multi-View Joint Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Project page: https://gohyojun15.github.io/VideoRFSplat/
    </div>
    <details class="paper-abstract">
      We propose VideoRFSplat, a direct text-to-3D model leveraging a video generation model to generate realistic 3D Gaussian Splatting (3DGS) for unbounded real-world scenes. To generate diverse camera poses and unbounded spatial extent of real-world scenes, while ensuring generalization to arbitrary text prompts, previous methods fine-tune 2D generative models to jointly model camera poses and multi-view images. However, these methods suffer from instability when extending 2D generative models to joint modeling due to the modality gap, which necessitates additional models to stabilize training and inference. In this work, we propose an architecture and a sampling strategy to jointly model multi-view images and camera poses when fine-tuning a video generation model. Our core idea is a dual-stream architecture that attaches a dedicated pose generation model alongside a pre-trained video generation model via communication blocks, generating multi-view images and camera poses through separate streams. This design reduces interference between the pose and image modalities. Additionally, we propose an asynchronous sampling strategy that denoises camera poses faster than multi-view images, allowing rapidly denoised poses to condition multi-view generation, reducing mutual ambiguity and enhancing cross-modal consistency. Trained on multiple large-scale real-world datasets (RealEstate10K, MVImgNet, DL3DV-10K, ACID), VideoRFSplat outperforms existing text-to-3D direct generation methods that heavily depend on post-hoc refinement via score distillation sampling, achieving superior results without such refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15835v1">BARD-GS: Blur-Aware Reconstruction of Dynamic Scenes via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 CVPR2025. Project page at https://vulab-ai.github.io/BARD-GS/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown remarkable potential for static scene reconstruction, and recent advancements have extended its application to dynamic scenes. However, the quality of reconstructions depends heavily on high-quality input images and precise camera poses, which are not that trivial to fulfill in real-world scenarios. Capturing dynamic scenes with handheld monocular cameras, for instance, typically involves simultaneous movement of both the camera and objects within a single exposure. This combined motion frequently results in image blur that existing methods cannot adequately handle. To address these challenges, we introduce BARD-GS, a novel approach for robust dynamic scene reconstruction that effectively handles blurry inputs and imprecise camera poses. Our method comprises two main components: 1) camera motion deblurring and 2) object motion deblurring. By explicitly decomposing motion blur into camera motion blur and object motion blur and modeling them separately, we achieve significantly improved rendering results in dynamic regions. In addition, we collect a real-world motion blur dataset of dynamic scenes to evaluate our approach. Extensive experiments demonstrate that BARD-GS effectively reconstructs high-quality dynamic scenes under realistic conditions, significantly outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16747v1">SAGE: Semantic-Driven Adaptive Gaussian Splatting in Extended Reality</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has significantly improved the efficiency and realism of three-dimensional scene visualization in several applications, ranging from robotics to eXtended Reality (XR). This work presents SAGE (Semantic-Driven Adaptive Gaussian Splatting in Extended Reality), a novel framework designed to enhance the user experience by dynamically adapting the Level of Detail (LOD) of different 3DGS objects identified via a semantic segmentation. Experimental results demonstrate how SAGE effectively reduces memory and computational overhead while keeping a desired target visual quality, thus providing a powerful optimization for interactive XR applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01846v3">UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 https://aashishrai3799.github.io/uvgs
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated superior quality in modeling 3D objects and scenes. However, generating 3DGS remains challenging due to their discrete, unstructured, and permutation-invariant nature. In this work, we present a simple yet effective method to overcome these challenges. We utilize spherical mapping to transform 3DGS into a structured 2D representation, termed UVGS. UVGS can be viewed as multi-channel images, with feature dimensions as a concatenation of Gaussian attributes such as position, scale, color, opacity, and rotation. We further find that these heterogeneous features can be compressed into a lower-dimensional (e.g., 3-channel) shared feature space using a carefully designed multi-branch network. The compressed UVGS can be treated as typical RGB images. Remarkably, we discover that typical VAEs trained with latent diffusion models can directly generalize to this new representation without additional training. Our novel representation makes it effortless to leverage foundational 2D models, such as diffusion models, to directly model 3DGS. Additionally, one can simply increase the 2D UV resolution to accommodate more Gaussians, making UVGS a scalable solution compared to typical 3D backbones. This approach immediately unlocks various novel generation applications of 3DGS by inherently utilizing the already developed superior 2D generation capabilities. In our experiments, we demonstrate various unconditional, conditional generation, and inpainting applications of 3DGS based on diffusion models, which were previously non-trivial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16710v1">4D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Simultaneously localizing camera poses and constructing Gaussian radiance fields in dynamic scenes establish a crucial bridge between 2D images and the 4D real world. Instead of removing dynamic objects as distractors and reconstructing only static environments, this paper proposes an efficient architecture that incrementally tracks camera poses and establishes the 4D Gaussian radiance fields in unknown scenarios by using a sequence of RGB-D images. First, by generating motion masks, we obtain static and dynamic priors for each pixel. To eliminate the influence of static scenes and improve the efficiency on learning the motion of dynamic objects, we classify the Gaussian primitives into static and dynamic Gaussian sets, while the sparse control points along with an MLP is utilized to model the transformation fields of the dynamic Gaussians. To more accurately learn the motion of dynamic Gaussians, a novel 2D optical flow map reconstruction algorithm is designed to render optical flows of dynamic objects between neighbor images, which are further used to supervise the 4D Gaussian radiance fields along with traditional photometric and geometric constraints. In experiments, qualitative and quantitative evaluation results show that the proposed method achieves robust tracking and high-quality view synthesis performance in real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16681v1">GauRast: Enhancing GPU Triangle Rasterizers to Accelerate 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 DAC 2025
    </div>
    <details class="paper-abstract">
      3D intelligence leverages rich 3D features and stands as a promising frontier in AI, with 3D rendering fundamental to many downstream applications. 3D Gaussian Splatting (3DGS), an emerging high-quality 3D rendering method, requires significant computation, making real-time execution on existing GPU-equipped edge devices infeasible. Previous efforts to accelerate 3DGS rely on dedicated accelerators that require substantial integration overhead and hardware costs. This work proposes an acceleration strategy that leverages the similarities between the 3DGS pipeline and the highly optimized conventional graphics pipeline in modern GPUs. Instead of developing a dedicated accelerator, we enhance existing GPU rasterizer hardware to efficiently support 3DGS operations. Our results demonstrate a 23$\times$ increase in processing speed and a 24$\times$ reduction in energy consumption, with improvements yielding 6$\times$ faster end-to-end runtime for the original 3DGS algorithm and 4$\times$ for the latest efficiency-improved pipeline, achieving 24 FPS and 46 FPS respectively. These enhancements incur only a minimal area overhead of 0.2\% relative to the entire SoC chip area, underscoring the practicality and efficiency of our approach for enabling 3DGS rendering on resource-constrained platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16747v1">SAGE: Semantic-Driven Adaptive Gaussian Splatting in Extended Reality</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has significantly improved the efficiency and realism of three-dimensional scene visualization in several applications, ranging from robotics to eXtended Reality (XR). This work presents SAGE (Semantic-Driven Adaptive Gaussian Splatting in Extended Reality), a novel framework designed to enhance the user experience by dynamically adapting the Level of Detail (LOD) of different 3DGS objects identified via a semantic segmentation. Experimental results demonstrate how SAGE effectively reduces memory and computational overhead while keeping a desired target visual quality, thus providing a powerful optimization for interactive XR applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05040v2">GaussRender: Learning 3D Occupancy with Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Understanding the 3D geometry and semantics of driving scenes is critical for safe autonomous driving. Recent advances in 3D occupancy prediction have improved scene representation but often suffer from spatial inconsistencies, leading to floating artifacts and poor surface localization. Existing voxel-wise losses (e.g., cross-entropy) fail to enforce geometric coherence. In this paper, we propose GaussRender, a module that improves 3D occupancy learning by enforcing projective consistency. Our key idea is to project both predicted and ground-truth 3D occupancy into 2D camera views, where we apply supervision. Our method penalizes 3D configurations that produce inconsistent 2D projections, thereby enforcing a more coherent 3D structure. To achieve this efficiently, we leverage differentiable rendering with Gaussian splatting. GaussRender seamlessly integrates with existing architectures while maintaining efficiency and requiring no inference-time modifications. Extensive evaluations on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate that GaussRender significantly improves geometric fidelity across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), achieving state-of-the-art results, particularly on surface-sensitive metrics. The code is open-sourced at https://github.com/valeoai/GaussRender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08352v2">Mitigating Ambiguities in 3D Classification with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      3D classification with point cloud input is a fundamental problem in 3D vision. However, due to the discrete nature and the insufficient material description of point cloud representations, there are ambiguities in distinguishing wire-like and flat surfaces, as well as transparent or reflective objects. To address these issues, we propose Gaussian Splatting (GS) point cloud-based 3D classification. We find that the scale and rotation coefficients in the GS point cloud help characterize surface types. Specifically, wire-like surfaces consist of multiple slender Gaussian ellipsoids, while flat surfaces are composed of a few flat Gaussian ellipsoids. Additionally, the opacity in the GS point cloud represents the transparency characteristics of objects. As a result, ambiguities in point cloud-based 3D classification can be mitigated utilizing GS point cloud as input. To verify the effectiveness of GS point cloud input, we construct the first real-world GS point cloud dataset in the community, which includes 20 categories with 200 objects in each category. Experiments not only validate the superiority of GS point cloud input, especially in distinguishing ambiguous objects, but also demonstrate the generalization ability across different classification methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16898v2">MonoGSDF: Exploring Monocular Geometric Cues for Gaussian Splatting-Guided Implicit Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Accurate meshing from monocular images remains a key challenge in 3D vision. While state-of-the-art 3D Gaussian Splatting (3DGS) methods excel at synthesizing photorealistic novel views through rasterization-based rendering, their reliance on sparse, explicit primitives severely limits their ability to recover watertight and topologically consistent 3D surfaces.We introduce MonoGSDF, a novel method that couples Gaussian-based primitives with a neural Signed Distance Field (SDF) for high-quality reconstruction. During training, the SDF guides Gaussians' spatial distribution, while at inference, Gaussians serve as priors to reconstruct surfaces, eliminating the need for memory-intensive Marching Cubes. To handle arbitrary-scale scenes, we propose a scaling strategy for robust generalization. A multi-resolution training scheme further refines details and monocular geometric cues from off-the-shelf estimators enhance reconstruction quality. Experiments on real-world datasets show MonoGSDF outperforms prior methods while maintaining efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19459v2">ArtGS: Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Building articulated objects is a key challenge in computer vision. Existing methods often fail to effectively integrate information across different object states, limiting the accuracy of part-mesh reconstruction and part dynamics modeling, particularly for complex multi-part articulated objects. We introduce ArtGS, a novel approach that leverages 3D Gaussians as a flexible and efficient representation to address these issues. Our method incorporates canonical Gaussians with coarse-to-fine initialization and updates for aligning articulated part information across different object states, and employs a skinning-inspired part dynamics modeling module to improve both part-mesh reconstruction and articulation learning. Extensive experiments on both synthetic and real-world datasets, including a new benchmark for complex multi-part objects, demonstrate that ArtGS achieves state-of-the-art performance in joint parameter estimation and part mesh reconstruction. Our approach significantly improves reconstruction quality and efficiency, especially for multi-part articulated objects. Additionally, we provide comprehensive analyses of our design choices, validating the effectiveness of each component to highlight potential areas for future improvement. Our work is made publicly available at: https://articulate-gs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.09868v2">SAFER-Splat: A Control Barrier Function for Safe Navigation with Online Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted to International Conference on Robotics and Automation
    </div>
    <details class="paper-abstract">
      SAFER-Splat (Simultaneous Action Filtering and Environment Reconstruction) is a real-time, scalable, and minimally invasive action filter, based on control barrier functions, for safe robotic navigation in a detailed map constructed at runtime using Gaussian Splatting (GSplat). We propose a novel Control Barrier Function (CBF) that not only induces safety with respect to all Gaussian primitives in the scene, but when synthesized into a controller, is capable of processing hundreds of thousands of Gaussians while maintaining a minimal memory footprint and operating at 15 Hz during online Splat training. Of the total compute time, a small fraction of it consumes GPU resources, enabling uninterrupted training. The safety layer is minimally invasive, correcting robot actions only when they are unsafe. To showcase the safety filter, we also introduce SplatBridge, an open-source software package built with ROS for real-time GSplat mapping for robots. We demonstrate the safety and robustness of our pipeline first in simulation, where our method is 20-50x faster, safer, and less conservative than competing methods based on neural radiance fields. Further, we demonstrate simultaneous GSplat mapping and safety filtering on a drone hardware platform using only on-board perception. We verify that under teleoperation a human pilot cannot invoke a collision. Our videos and codebase can be found at https://chengine.github.io/safer-splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14475v1">Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency Modulation</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      The field of Novel View Synthesis has been revolutionized by 3D Gaussian Splatting (3DGS), which enables high-quality scene reconstruction that can be rendered in real-time. 3DGS-based techniques typically suffer from high GPU memory and disk storage requirements which limits their practical application on consumer-grade devices. We propose Opti3DGS, a novel frequency-modulated coarse-to-fine optimization framework that aims to minimize the number of Gaussian primitives used to represent a scene, thus reducing memory and storage demands. Opti3DGS leverages image frequency modulation, initially enforcing a coarse scene representation and progressively refining it by modulating frequency details in the training images. On the baseline 3DGS, we demonstrate an average reduction of 62% in Gaussians, a 40% reduction in the training GPU memory requirements and a 20% reduction in optimization time without sacrificing the visual quality. Furthermore, we show that our method integrates seamlessly with many 3DGS-based techniques, consistently reducing the number of Gaussian primitives while maintaining, and often improving, visual quality. Additionally, Opti3DGS inherently produces a level-of-detail scene representation at no extra cost, a natural byproduct of the optimization pipeline. Results and code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02009v2">Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Exploring real-world spaces using novel-view synthesis is fun, and reimagining those worlds in a different style adds another layer of excitement. Stylized worlds can also be used for downstream tasks where there is limited training data and a need to expand a model's training distribution. Most current novel-view synthesis stylization techniques lack the ability to convincingly change geometry. This is because any geometry change requires increased style strength which is often capped for stylization stability and consistency. In this work, we propose a new autoregressive 3D Gaussian Splatting stylization method. As part of this method, we contribute a new RGBD diffusion model that allows for strength control over appearance and shape stylization. To ensure consistency across stylized frames, we use a combination of novel depth-guided cross attention, feature injection, and a Warp ControlNet conditioned on composite frames for guiding the stylization of new frames. We validate our method via extensive qualitative results, quantitative experiments, and a user study. Code online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14274v1">Improving Adaptive Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has become one of the most influential works in the past year. Due to its efficient and high-quality novel view synthesis capabilities, it has been widely adopted in many research fields and applications. Nevertheless, 3DGS still faces challenges to properly manage the number of Gaussian primitives that are used during scene reconstruction. Following the adaptive density control (ADC) mechanism of 3D Gaussian Splatting, new Gaussians in under-reconstructed regions are created, while Gaussians that do not contribute to the rendering quality are pruned. We observe that those criteria for densifying and pruning Gaussians can sometimes lead to worse rendering by introducing artifacts. We especially observe under-reconstructed background or overfitted foreground regions. To encounter both problems, we propose three new improvements to the adaptive density control mechanism. Those include a correction for the scene extent calculation that does not only rely on camera positions, an exponentially ascending gradient threshold to improve training convergence, and significance-aware pruning strategy to avoid background artifacts. With these adaptions, we show that the rendering quality improves while using the same number of Gaussians primitives. Furthermore, with our improvements, the training converges considerably faster, allowing for more than twice as fast training times while yielding better quality than 3DGS. Finally, our contributions are easily compatible with most existing derivative works of 3DGS making them relevant for future works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14198v1">RoGSplat: Learning Robust Generalizable Human Gaussian Splatting from Sparse Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Accepted to CVPR2025
    </div>
    <details class="paper-abstract">
      This paper presents RoGSplat, a novel approach for synthesizing high-fidelity novel views of unseen human from sparse multi-view images, while requiring no cumbersome per-subject optimization. Unlike previous methods that typically struggle with sparse views with few overlappings and are less effective in reconstructing complex human geometry, the proposed method enables robust reconstruction in such challenging conditions. Our key idea is to lift SMPL vertices to dense and reliable 3D prior points representing accurate human body geometry, and then regress human Gaussian parameters based on the points. To account for possible misalignment between SMPL model and images, we propose to predict image-aligned 3D prior points by leveraging both pixel-level features and voxel-level features, from which we regress the coarse Gaussians. To enhance the ability to capture high-frequency details, we further render depth maps from the coarse 3D Gaussians to help regress fine-grained pixel-wise Gaussians. Experiments on several benchmark datasets demonstrate that our method outperforms state-of-the-art methods in novel view synthesis and cross-dataset generalization. Our code is available at https://github.com/iSEE-Laboratory/RoGSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14171v1">Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      We introduce an image upscaling technique tailored for 3D Gaussian Splatting (3DGS) on lightweight GPUs. Compared to 3DGS, it achieves significantly higher rendering speeds and reduces artifacts commonly observed in 3DGS reconstructions. Our technique upscales low-resolution 3DGS renderings with a marginal increase in cost by directly leveraging the analytical image gradients of Gaussians for gradient-based bicubic spline interpolation. The technique is agnostic to the specific 3DGS implementation, achieving novel view synthesis at rates 3x-4x higher than the baseline implementation. Through extensive experiments on multiple datasets, we showcase the performance improvements and high reconstruction fidelity attainable with gradient-aware upscaling of 3DGS images. We further demonstrate the integration of gradient-aware upscaling into the gradient-based optimization of a 3DGS model and analyze its effects on reconstruction quality and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04013v2">3D-LMVIC: Learning-based Multi-View Image Coding with 3D Gaussian Geometric Priors</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 17 pages, 10 figures, conference
    </div>
    <details class="paper-abstract">
      Existing multi-view image compression methods often rely on 2D projection-based similarities between views to estimate disparities. While effective for small disparities, such as those in stereo images, these methods struggle with the more complex disparities encountered in wide-baseline multi-camera systems, commonly found in virtual reality and autonomous driving applications. To address this limitation, we propose 3D-LMVIC, a novel learning-based multi-view image compression framework that leverages 3D Gaussian Splatting to derive geometric priors for accurate disparity estimation. Furthermore, we introduce a depth map compression model to minimize geometric redundancy across views, along with a multi-view sequence ordering strategy based on a defined distance measure between views to enhance correlations between adjacent views. Experimental results demonstrate that 3D-LMVIC achieves superior performance compared to both traditional and learning-based methods. Additionally, it significantly improves disparity estimation accuracy over existing two-view approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14029v1">Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 CVPR 2025. The code is publicly available at this https URL (https://github.com/Runsong123/Unified-Lift)
    </div>
    <details class="paper-abstract">
      Lifting multi-view 2D instance segmentation to a radiance field has proven to be effective to enhance 3D understanding. Existing methods rely on direct matching for end-to-end lifting, yielding inferior results; or employ a two-stage solution constrained by complex pre- or post-processing. In this work, we design a new end-to-end object-aware lifting approach, named Unified-Lift that provides accurate 3D segmentation based on the 3D Gaussian representation. To start, we augment each Gaussian point with an additional Gaussian-level feature learned using a contrastive loss to encode instance information. Importantly, we introduce a learnable object-level codebook to account for individual objects in the scene for an explicit object-level understanding and associate the encoded object-level features with the Gaussian-level point features for segmentation predictions. While promising, achieving effective codebook learning is non-trivial and a naive solution leads to degraded performance. Therefore, we formulate the association learning module and the noisy label filtering module for effective and robust codebook learning. We conduct experiments on three benchmarks: LERF-Masked, Replica, and Messy Rooms datasets. Both qualitative and quantitative results manifest that our Unified-Lift clearly outperforms existing methods in terms of segmentation quality and time efficiency. The code is publicly available at \href{https://github.com/Runsong123/Unified-Lift}{https://github.com/Runsong123/Unified-Lift}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12335v2">GS-I$^{3}$: Gaussian Splatting for Surface Reconstruction from Illumination-Inconsistent Images</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Comments: This work has been submitted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) for possible publication
    </div>
    <details class="paper-abstract">
      Accurate geometric surface reconstruction, providing essential environmental information for navigation and manipulation tasks, is critical for enabling robotic self-exploration and interaction. Recently, 3D Gaussian Splatting (3DGS) has gained significant attention in the field of surface reconstruction due to its impressive geometric quality and computational efficiency. While recent relevant advancements in novel view synthesis under inconsistent illumination using 3DGS have shown promise, the challenge of robust surface reconstruction under such conditions is still being explored. To address this challenge, we propose a method called GS-3I. Specifically, to mitigate 3D Gaussian optimization bias caused by underexposed regions in single-view images, based on Convolutional Neural Network (CNN), a tone mapping correction framework is introduced. Furthermore, inconsistent lighting across multi-view images, resulting from variations in camera settings and complex scene illumination, often leads to geometric constraint mismatches and deviations in the reconstructed surface. To overcome this, we propose a normal compensation mechanism that integrates reference normals extracted from single-view image with normals computed from multi-view observations to effectively constrain geometric inconsistencies. Extensive experimental evaluations demonstrate that GS-3I can achieve robust and accurate surface reconstruction across complex illumination scenarios, highlighting its effectiveness and versatility in this critical challenge. https://github.com/TFwang-9527/GS-3I
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13948v1">Light4GS: Lightweight Compact 4D Gaussian Splatting Generation via Context Model</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as an efficient and high-fidelity paradigm for novel view synthesis. To adapt 3DGS for dynamic content, deformable 3DGS incorporates temporally deformable primitives with learnable latent embeddings to capture complex motions. Despite its impressive performance, the high-dimensional embeddings and vast number of primitives lead to substantial storage requirements. In this paper, we introduce a \textbf{Light}weight \textbf{4}D\textbf{GS} framework, called Light4GS, that employs significance pruning with a deep context model to provide a lightweight storage-efficient dynamic 3DGS representation. The proposed Light4GS is based on 4DGS that is a typical representation of deformable 3DGS. Specifically, our framework is built upon two core components: (1) a spatio-temporal significance pruning strategy that eliminates over 64\% of the deformable primitives, followed by an entropy-constrained spherical harmonics compression applied to the remainder; and (2) a deep context model that integrates intra- and inter-prediction with hyperprior into a coarse-to-fine context structure to enable efficient multiscale latent embedding compression. Our approach achieves over 120x compression and increases rendering FPS up to 20\% compared to the baseline 4DGS, and also superior to frame-wise state-of-the-art 3DGS compression methods, revealing the effectiveness of our Light4GS in terms of both intra- and inter-prediction methods without sacrificing rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12001v2">3D Gaussian Splatting against Moving Objects for High-Fidelity Street Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      The accurate reconstruction of dynamic street scenes is critical for applications in autonomous driving, augmented reality, and virtual reality. Traditional methods relying on dense point clouds and triangular meshes struggle with moving objects, occlusions, and real-time processing constraints, limiting their effectiveness in complex urban environments. While multi-view stereo and neural radiance fields have advanced 3D reconstruction, they face challenges in computational efficiency and handling scene dynamics. This paper proposes a novel 3D Gaussian point distribution method for dynamic street scene reconstruction. Our approach introduces an adaptive transparency mechanism that eliminates moving objects while preserving high-fidelity static scene details. Additionally, iterative refinement of Gaussian point distribution enhances geometric accuracy and texture representation. We integrate directional encoding with spatial position optimization to optimize storage and rendering efficiency, reducing redundancy while maintaining scene integrity. Experimental results demonstrate that our method achieves high reconstruction quality, improved rendering performance, and adaptability in large-scale dynamic environments. These contributions establish a robust framework for real-time, high-precision 3D reconstruction, advancing the practicality of dynamic scene modeling across multiple applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14736v1">HandSplat: Embedding-Driven Gaussian Splatting for High-Fidelity Hand Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Existing 3D Gaussian Splatting (3DGS) methods for hand rendering rely on rigid skeletal motion with an oversimplified non-rigid motion model, which fails to capture fine geometric and appearance details. Additionally, they perform densification based solely on per-point gradients and process poses independently, ignoring spatial and temporal correlations. These limitations lead to geometric detail loss, temporal instability, and inefficient point distribution. To address these issues, we propose HandSplat, a novel Gaussian Splatting-based framework that enhances both fidelity and stability for hand rendering. To improve fidelity, we extend standard 3DGS attributes with implicit geometry and appearance embeddings for finer non-rigid motion modeling while preserving the static hand characteristic modeled by original 3DGS attributes. Additionally, we introduce a local gradient-aware densification strategy that dynamically refines Gaussian density in high-variation regions. To improve stability, we incorporate pose-conditioned attribute regularization to encourage attribute consistency across similar poses, mitigating temporal artifacts. Extensive experiments on InterHand2.6M demonstrate that HandSplat surpasses existing methods in fidelity and stability while achieving real-time performance. We will release the code and pre-trained models upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14698v1">SplatVoxel: History-Aware Novel View Streaming without Temporal Training</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      We study the problem of novel view streaming from sparse-view videos, which aims to generate a continuous sequence of high-quality, temporally consistent novel views as new input frames arrive. However, existing novel view synthesis methods struggle with temporal coherence and visual fidelity, leading to flickering and inconsistency. To address these challenges, we introduce history-awareness, leveraging previous frames to reconstruct the scene and improve quality and stability. We propose a hybrid splat-voxel feed-forward scene reconstruction approach that combines Gaussian Splatting to propagate information over time, with a hierarchical voxel grid for temporal fusion. Gaussian primitives are efficiently warped over time using a motion graph that extends 2D tracking models to 3D motion, while a sparse voxel transformer integrates new temporal observations in an error-aware manner. Crucially, our method does not require training on multi-view video datasets, which are currently limited in size and diversity, and can be directly applied to sparse-view video streams in a history-aware manner at inference time. Our approach achieves state-of-the-art performance in both static and streaming scene reconstruction, effectively reducing temporal artifacts and visual artifacts while running at interactive rates (15 fps with 350ms delay) on a single H100 GPU. Project Page: https://19reborn.github.io/SplatVoxel/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2312.00846v2">NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Existing neural implicit surface reconstruction methods have achieved impressive performance in multi-view 3D reconstruction by leveraging explicit geometry priors such as depth maps or point clouds as regularization. However, the reconstruction results still lack fine details because of the over-smoothed depth map or sparse point cloud. In this work, we propose a neural implicit surface reconstruction pipeline with guidance from 3D Gaussian Splatting to recover highly detailed surfaces. The advantage of 3D Gaussian Splatting is that it can generate dense point clouds with detailed structure. Nonetheless, a naive adoption of 3D Gaussian Splatting can fail since the generated points are the centers of 3D Gaussians that do not necessarily lie on the surface. We thus introduce a scale regularizer to pull the centers close to the surface by enforcing the 3D Gaussians to be extremely thin. Moreover, we propose to refine the point cloud from 3D Gaussians Splatting with the normal priors from the surface predicted by neural implicit models instead of using a fixed set of points as guidance. Consequently, the quality of surface reconstruction improves from the guidance of the more accurate 3D Gaussian splatting. By jointly optimizing the 3D Gaussian Splatting and the neural implicit model, our approach benefits from both representations and generates complete surfaces with intricate details. Experiments on Tanks and Temples verify the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2503.04082v2">Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Real2Sim is becoming increasingly important with the rapid development of surgical artificial intelligence (AI) and autonomy. In this work, we propose a novel Real2Sim methodology, Instrument-Splatting, that leverages 3D Gaussian Splatting to provide fully controllable 3D reconstruction of surgical instruments from monocular surgical videos. To maintain both high visual fidelity and manipulability, we introduce a geometry pre-training to bind Gaussian point clouds on part mesh with accurate geometric priors and define a forward kinematics to control the Gaussians as flexible as real instruments. Afterward, to handle unposed videos, we design a novel instrument pose tracking method leveraging semantics-embedded Gaussians to robustly refine per-frame instrument poses and joint states in a render-and-compare manner, which allows our instrument Gaussian to accurately learn textures and reach photorealistic rendering. We validated our method on 2 publicly released surgical videos and 4 videos collected on ex vivo tissues and green screens. Quantitative and qualitative evaluations demonstrate the effectiveness and superiority of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19895v5">GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 This paper is accepted by the IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR), 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently created impressive 3D assets for various applications. However, considering security, capacity, invisibility, and training efficiency, the copyright of 3DGS assets is not well protected as existing watermarking methods are unsuited for its rendering pipeline. In this paper, we propose GuardSplat, an innovative and efficient framework for watermarking 3DGS assets. Specifically, 1) We propose a CLIP-guided pipeline for optimizing the message decoder with minimal costs. The key objective is to achieve high-accuracy extraction by leveraging CLIP's aligning capability and rich representations, demonstrating exceptional capacity and efficiency. 2) We tailor a Spherical-Harmonic-aware (SH-aware) Message Embedding module for 3DGS, seamlessly embedding messages into the SH features of each 3D Gaussian while preserving the original 3D structure. This enables watermarking 3DGS assets with minimal fidelity trade-offs and prevents malicious users from removing the watermarks from the model files, meeting the demands for invisibility and security. 3) We present an Anti-distortion Message Extraction module to improve robustness against various distortions. Experiments demonstrate that GuardSplat outperforms state-of-the-art and achieves fast optimization speed. Project page is at https://narcissusex.github.io/GuardSplat, and Code is at https://github.com/NarcissusEx/GuardSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13272v1">Generative Gaussian Splatting: Generating 3D Scenes with Video Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Synthesizing consistent and photorealistic 3D scenes is an open problem in computer vision. Video diffusion models generate impressive videos but cannot directly synthesize 3D representations, i.e., lack 3D consistency in the generated sequences. In addition, directly training generative 3D models is challenging due to a lack of 3D training data at scale. In this work, we present Generative Gaussian Splatting (GGS) -- a novel approach that integrates a 3D representation with a pre-trained latent video diffusion model. Specifically, our model synthesizes a feature field parameterized via 3D Gaussian primitives. The feature field is then either rendered to feature maps and decoded into multi-view images, or directly upsampled into a 3D radiance field. We evaluate our approach on two common benchmark datasets for scene synthesis, RealEstate10K and ScanNet+, and find that our proposed GGS model significantly improves both the 3D consistency of the generated multi-view images, and the quality of the generated 3D scenes over all relevant baselines. Compared to a similar model without 3D representation, GGS improves FID on the generated 3D scenes by ~20% on both RealEstate10K and ScanNet+. Project page: https://katjaschwarz.github.io/ggs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13176v1">DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Reconstructing clean, distractor-free 3D scenes from real-world captures remains a significant challenge, particularly in highly dynamic and cluttered settings such as egocentric videos. To tackle this problem, we introduce DeGauss, a simple and robust self-supervised framework for dynamic scene reconstruction based on a decoupled dynamic-static Gaussian Splatting design. DeGauss models dynamic elements with foreground Gaussians and static content with background Gaussians, using a probabilistic mask to coordinate their composition and enable independent yet complementary optimization. DeGauss generalizes robustly across a wide range of real-world scenarios, from casual image collections to long, dynamic egocentric videos, without relying on complex heuristics or extensive supervision. Experiments on benchmarks including NeRF-on-the-go, ADT, AEA, Hot3D, and EPIC-Fields demonstrate that DeGauss consistently outperforms existing methods, establishing a strong baseline for generalizable, distractor-free 3D reconstructionin highly dynamic, interaction-rich environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13086v1">Gaussian On-the-Fly Splatting: A Progressive Framework for Robust Near Real-Time 3DGS Optimization</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) achieves high-fidelity rendering with fast real-time performance, but existing methods rely on offline training after full Structure-from-Motion (SfM) processing. In contrast, this work introduces On-the-Fly GS, a progressive framework enabling near real-time 3DGS optimization during image capture. As each image arrives, its pose and sparse points are updated via on-the-fly SfM, and newly optimized Gaussians are immediately integrated into the 3DGS field. We propose a progressive local optimization strategy to prioritize new images and their neighbors by their corresponding overlapping relationship, allowing the new image and its overlapping images to get more training. To further stabilize training across old and new images, an adaptive learning rate schedule balances the iterations and the learning rate. Moreover, to maintain overall quality of the 3DGS field, an efficient global optimization scheme prevents overfitting to the newly added images. Experiments on multiple benchmark datasets show that our On-the-Fly GS reduces training time significantly, optimizing each new image in seconds with minimal rendering loss, offering the first practical step toward rapid, progressive 3DGS reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12663v2">LAGA: Layered 3D Avatar Generation and Customization via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Creating and customizing a 3D clothed avatar from textual descriptions is a critical and challenging task. Traditional methods often treat the human body and clothing as inseparable, limiting users' ability to freely mix and match garments. In response to this limitation, we present LAyered Gaussian Avatar (LAGA), a carefully designed framework enabling the creation of high-fidelity decomposable avatars with diverse garments. By decoupling garments from avatar, our framework empowers users to conviniently edit avatars at the garment level. Our approach begins by modeling the avatar using a set of Gaussian points organized in a layered structure, where each layer corresponds to a specific garment or the human body itself. To generate high-quality garments for each layer, we introduce a coarse-to-fine strategy for diverse garment generation and a novel dual-SDS loss function to maintain coherence between the generated garments and avatar components, including the human body and other garments. Moreover, we introduce three regularization losses to guide the movement of Gaussians for garment transfer, allowing garments to be freely transferred to various avatars. Extensive experimentation demonstrates that our approach surpasses existing methods in the generation of 3D clothed humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03714v2">MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/MoDecGS-site/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in scene representation and neural rendering, with intense efforts focused on adapting it for dynamic scenes. Despite delivering remarkable rendering quality and speed, existing methods struggle with storage demands and representing complex real-world motions. To tackle these issues, we propose MoDecGS, a memory-efficient Gaussian splatting framework designed for reconstructing novel views in challenging scenarios with complex motions. We introduce GlobaltoLocal Motion Decomposition (GLMD) to effectively capture dynamic motions in a coarsetofine manner. This approach leverages Global Canonical Scaffolds (Global CS) and Local Canonical Scaffolds (Local CS), extending static Scaffold representation to dynamic video reconstruction. For Global CS, we propose Global Anchor Deformation (GAD) to efficiently represent global dynamics along complex motions, by directly deforming the implicit Scaffold attributes which are anchor position, offset, and local context features. Next, we finely adjust local motions via the Local Gaussian Deformation (LGD) of Local CS explicitly. Additionally, we introduce Temporal Interval Adjustment (TIA) to automatically control the temporal coverage of each Local CS during training, allowing MoDecGS to find optimal interval assignments based on the specified number of temporal segments. Extensive evaluations demonstrate that MoDecGS achieves an average 70% reduction in model size over stateoftheart methods for dynamic 3D Gaussians from realworld dynamic videos while maintaining or even improving rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00846v2">NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Existing neural implicit surface reconstruction methods have achieved impressive performance in multi-view 3D reconstruction by leveraging explicit geometry priors such as depth maps or point clouds as regularization. However, the reconstruction results still lack fine details because of the over-smoothed depth map or sparse point cloud. In this work, we propose a neural implicit surface reconstruction pipeline with guidance from 3D Gaussian Splatting to recover highly detailed surfaces. The advantage of 3D Gaussian Splatting is that it can generate dense point clouds with detailed structure. Nonetheless, a naive adoption of 3D Gaussian Splatting can fail since the generated points are the centers of 3D Gaussians that do not necessarily lie on the surface. We thus introduce a scale regularizer to pull the centers close to the surface by enforcing the 3D Gaussians to be extremely thin. Moreover, we propose to refine the point cloud from 3D Gaussians Splatting with the normal priors from the surface predicted by neural implicit models instead of using a fixed set of points as guidance. Consequently, the quality of surface reconstruction improves from the guidance of the more accurate 3D Gaussian splatting. By jointly optimizing the 3D Gaussian Splatting and the neural implicit model, our approach benefits from both representations and generates complete surfaces with intricate details. Experiments on Tanks and Temples verify the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04459v3">Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 CVPR 2025; Project page at https://svraster.github.io/ ; Code at https://github.com/NVlabs/svraster
    </div>
    <details class="paper-abstract">
      We propose an efficient radiance field rendering algorithm that incorporates a rasterization process on adaptive sparse voxels without neural networks or 3D Gaussians. There are two key contributions coupled with the proposed system. The first is to adaptively and explicitly allocate sparse voxels to different levels of detail within scenes, faithfully reproducing scene details with $65536^3$ grid resolution while achieving high rendering frame rates. Second, we customize a rasterizer for efficient adaptive sparse voxels rendering. We render voxels in the correct depth order by using ray direction-dependent Morton ordering, which avoids the well-known popping artifact found in Gaussian splatting. Our method improves the previous neural-free voxel model by over 4db PSNR and more than 10x FPS speedup, achieving state-of-the-art comparable novel-view synthesis results. Additionally, our voxel representation is seamlessly compatible with grid-based 3D processing techniques such as Volume Fusion, Voxel Pooling, and Marching Cubes, enabling a wide range of future extensions and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12862v1">CAT-3DGS Pro: A New Benchmark for Efficient 3DGS Compression</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown immense potential for novel view synthesis. However, achieving rate-distortion-optimized compression of 3DGS representations for transmission and/or storage applications remains a challenge. CAT-3DGS introduces a context-adaptive triplane hyperprior for end-to-end optimized compression, delivering state-of-the-art coding performance. Despite this, it requires prolonged training and decoding time. To address these limitations, we propose CAT-3DGS Pro, an enhanced version of CAT-3DGS that improves both compression performance and computational efficiency. First, we introduce a PCA-guided vector-matrix hyperprior, which replaces the triplane-based hyperprior to reduce redundant parameters. To achieve a more balanced rate-distortion trade-off and faster encoding, we propose an alternate optimization strategy (A-RDO). Additionally, we refine the sampling rate optimization method in CAT-3DGS, leading to significant improvements in rate-distortion performance. These enhancements result in a 46.6% BD-rate reduction and 3x speedup in training time on BungeeNeRF, while achieving 5x acceleration in decoding speed for the Amsterdam scene compared to CAT-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12836v1">CompMarkGS: Robust Watermarking for Compression 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 23 pages, 17 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) enables rapid differentiable rendering for 3D reconstruction and novel view synthesis, leading to its widespread commercial use. Consequently, copyright protection via watermarking has become critical. However, because 3DGS relies on millions of Gaussians, which require gigabytes of storage, efficient transfer and storage require compression. Existing 3DGS watermarking methods are vulnerable to quantization-based compression, often resulting in the loss of the embedded watermark. To address this challenge, we propose a novel watermarking method that ensures watermark robustness after model compression while maintaining high rendering quality. In detail, we incorporate a quantization distortion layer that simulates compression during training, preserving the watermark under quantization-based compression. Also, we propose a learnable watermark embedding feature that embeds the watermark into the anchor feature, ensuring structural consistency and seamless integration into the 3D scene. Furthermore, we present a frequency-aware anchor growing mechanism to enhance image quality in high-frequency regions by effectively identifying Guassians within these regions. Experimental results confirm that our method preserves the watermark and maintains superior image quality under high compression, validating it as a promising approach for a secure 3DGS model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12806v1">AV-Surf: Surface-Enhanced Geometry-Aware Novel-View Acoustic Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Accurately modeling sound propagation with complex real-world environments is essential for Novel View Acoustic Synthesis (NVAS). While previous studies have leveraged visual perception to estimate spatial acoustics, the combined use of surface normal and structural details from 3D representations in acoustic modeling has been underexplored. Given their direct impact on sound wave reflections and propagation, surface normals should be jointly modeled with structural details to achieve accurate spatial acoustics. In this paper, we propose a surface-enhanced geometry-aware approach for NVAS to improve spatial acoustic modeling. To achieve this, we exploit geometric priors, such as image, depth map, surface normals, and point clouds obtained using a 3D Gaussian Splatting (3DGS) based framework. We introduce a dual cross-attention-based transformer integrating geometrical constraints into frequency query to understand the surroundings of the emitter. Additionally, we design a ConvNeXt-based spectral features processing network called Spectral Refinement Network (SRN) to synthesize realistic binaural audio. Experimental results on the RWAVS and SoundSpace datasets highlight the necessity of our approach, as it surpasses existing methods in novel view acoustic synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08317v2">Uni-Gaussians: Unifying Camera and Lidar Simulation with Gaussians for Dynamic Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles necessitates comprehensive simulation of multi-sensor data, encompassing inputs from both cameras and LiDAR sensors, across various dynamic driving scenarios. Neural rendering techniques, which utilize collected raw sensor data to simulate these dynamic environments, have emerged as a leading methodology. While NeRF-based approaches can uniformly represent scenes for rendering data from both camera and LiDAR, they are hindered by slow rendering speeds due to dense sampling. Conversely, Gaussian Splatting-based methods employ Gaussian primitives for scene representation and achieve rapid rendering through rasterization. However, these rasterization-based techniques struggle to accurately model non-linear optical sensors. This limitation restricts their applicability to sensors beyond pinhole cameras. To address these challenges and enable unified representation of dynamic driving scenarios using Gaussian primitives, this study proposes a novel hybrid approach. Our method utilizes rasterization for rendering image data while employing Gaussian ray-tracing for LiDAR data rendering. Experimental results on public datasets demonstrate that our approach outperforms current state-of-the-art methods. This work presents a unified and efficient solution for realistic simulation of camera and LiDAR data in autonomous driving scenarios using Gaussian primitives, offering significant advancements in both rendering quality and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20309v5">InstantSplat: Sparse-view Gaussian Splatting in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Project Page: https://instantsplat.github.io/
    </div>
    <details class="paper-abstract">
      While neural 3D reconstruction has advanced substantially, its performance significantly degrades with sparse-view data, which limits its broader applicability, since SfM is often unreliable in sparse-view scenarios where feature matches are scarce. In this paper, we introduce InstantSplat, a novel approach for addressing sparse-view 3D scene reconstruction at lightning-fast speed. InstantSplat employs a self-supervised framework that optimizes 3D scene representation and camera poses by unprojecting 2D pixels into 3D space and aligning them using differentiable neural rendering. The optimization process is initialized with a large-scale trained geometric foundation model, which provides dense priors that yield initial points through model inference, after which we further optimize all scene parameters using photometric errors. To mitigate redundancy introduced by the prior model, we propose a co-visibility-based geometry initialization, and a Gaussian-based bundle adjustment is employed to rapidly adapt both the scene representation and camera parameters without relying on a complex adaptive density control process. Overall, InstantSplat is compatible with multiple point-based representations for view synthesis and surface reconstruction. It achieves an acceleration of over 30x in reconstruction and improves visual quality (SSIM) from 0.3755 to 0.7624 compared to traditional SfM with 3D-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09868v2">SAFER-Splat: A Control Barrier Function for Safe Navigation with Online Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Accepted to International Conference on Robotics and Automation
    </div>
    <details class="paper-abstract">
      SAFER-Splat (Simultaneous Action Filtering and Environment Reconstruction) is a real-time, scalable, and minimally invasive action filter, based on control barrier functions, for safe robotic navigation in a detailed map constructed at runtime using Gaussian Splatting (GSplat). We propose a novel Control Barrier Function (CBF) that not only induces safety with respect to all Gaussian primitives in the scene, but when synthesized into a controller, is capable of processing hundreds of thousands of Gaussians while maintaining a minimal memory footprint and operating at 15 Hz during online Splat training. Of the total compute time, a small fraction of it consumes GPU resources, enabling uninterrupted training. The safety layer is minimally invasive, correcting robot actions only when they are unsafe. To showcase the safety filter, we also introduce SplatBridge, an open-source software package built with ROS for real-time GSplat mapping for robots. We demonstrate the safety and robustness of our pipeline first in simulation, where our method is 20-50x faster, safer, and less conservative than competing methods based on neural radiance fields. Further, we demonstrate simultaneous GSplat mapping and safety filtering on a drone hardware platform using only on-board perception. We verify that under teleoperation a human pilot cannot invoke a collision. Our videos and codebase can be found at https://chengine.github.io/safer-splat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00846v2">NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Existing neural implicit surface reconstruction methods have achieved impressive performance in multi-view 3D reconstruction by leveraging explicit geometry priors such as depth maps or point clouds as regularization. However, the reconstruction results still lack fine details because of the over-smoothed depth map or sparse point cloud. In this work, we propose a neural implicit surface reconstruction pipeline with guidance from 3D Gaussian Splatting to recover highly detailed surfaces. The advantage of 3D Gaussian Splatting is that it can generate dense point clouds with detailed structure. Nonetheless, a naive adoption of 3D Gaussian Splatting can fail since the generated points are the centers of 3D Gaussians that do not necessarily lie on the surface. We thus introduce a scale regularizer to pull the centers close to the surface by enforcing the 3D Gaussians to be extremely thin. Moreover, we propose to refine the point cloud from 3D Gaussians Splatting with the normal priors from the surface predicted by neural implicit models instead of using a fixed set of points as guidance. Consequently, the quality of surface reconstruction improves from the guidance of the more accurate 3D Gaussian splatting. By jointly optimizing the 3D Gaussian Splatting and the neural implicit model, our approach benefits from both representations and generates complete surfaces with intricate details. Experiments on Tanks and Temples verify the effectiveness of our proposed method.
    </details>
</div>
