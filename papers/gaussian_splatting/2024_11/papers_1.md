# gaussian splatting - 2024_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00578v1">Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives</a></div>
    <div class="paper-meta">
      📅 2024-11-30
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) is a recent 3D scene reconstruction technique that enables real-time rendering of novel views by modeling scenes as parametric point clouds of differentiable 3D Gaussians. However, its rendering speed and model size still present bottlenecks, especially in resource-constrained settings. In this paper, we identify and address two key inefficiencies in 3D-GS, achieving substantial improvements in rendering speed, model size, and training time. First, we optimize the rendering pipeline to precisely localize Gaussians in the scene, boosting rendering speed without altering visual fidelity. Second, we introduce a novel pruning technique and integrate it into the training pipeline, significantly reducing model size and training time while further raising rendering speed. Our Speedy-Splat approach combines these techniques to accelerate average rendering speed by a drastic $6.71\times$ across scenes from the Mip-NeRF 360, Tanks & Temples, and Deep Blending datasets with $10.6\times$ fewer primitives than 3D-GS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00518v1">Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 project page: https://amirbarda.github.io/Instant3dit.github.io/
    </div>
    <details class="paper-abstract">
      We propose a generative technique to edit 3D shapes, represented as meshes, NeRFs, or Gaussian Splats, in approximately 3 seconds, without the need for running an SDS type of optimization. Our key insight is to cast 3D editing as a multiview image inpainting problem, as this representation is generic and can be mapped back to any 3D representation using the bank of available Large Reconstruction Models. We explore different fine-tuning strategies to obtain both multiview generation and inpainting capabilities within the same diffusion model. In particular, the design of the inpainting mask is an important factor of training an inpainting model, and we propose several masking strategies to mimic the types of edits a user would perform on a 3D shape. Our approach takes 3D generative editing from hours to seconds and produces higher-quality results compared to previous works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16767v2">ReconX: Reconstruct Any Scene from Sparse Views with Video Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 Project page: https://liuff19.github.io/ReconX
    </div>
    <details class="paper-abstract">
      Advancements in 3D scene reconstruction have transformed 2D images from the real world into 3D models, producing realistic 3D results from hundreds of input photos. Despite great success in dense-view reconstruction scenarios, rendering a detailed scene from insufficient captured views is still an ill-posed optimization problem, often resulting in artifacts and distortions in unseen areas. In this paper, we propose ReconX, a novel 3D scene reconstruction paradigm that reframes the ambiguous reconstruction challenge as a temporal generation task. The key insight is to unleash the strong generative prior of large pre-trained video diffusion models for sparse-view reconstruction. However, 3D view consistency struggles to be accurately preserved in directly generated video frames from pre-trained models. To address this, given limited input views, the proposed ReconX first constructs a global point cloud and encodes it into a contextual space as the 3D structure condition. Guided by the condition, the video diffusion model then synthesizes video frames that are both detail-preserved and exhibit a high degree of 3D consistency, ensuring the coherence of the scene from various perspectives. Finally, we recover the 3D scene from the generated video through a confidence-aware 3D Gaussian Splatting optimization scheme. Extensive experiments on various real-world datasets show the superiority of our ReconX over state-of-the-art methods in terms of quality and generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00392v1">GradiSeg: Gradient-Guided Gaussian Segmentation with Enhanced 3D Boundary Precision</a></div>
    <div class="paper-meta">
      📅 2024-11-30
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting enables high-quality real-time rendering, existing Gaussian-based frameworks for 3D semantic segmentation still face significant challenges in boundary recognition accuracy. To address this, we propose a novel 3DGS-based framework named GradiSeg, incorporating Identity Encoding to construct a deeper semantic understanding of scenes. Our approach introduces two key modules: Identity Gradient Guided Densification (IGD) and Local Adaptive K-Nearest Neighbors (LA-KNN). The IGD module supervises gradients of Identity Encoding to refine Gaussian distributions along object boundaries, aligning them closely with boundary contours. Meanwhile, the LA-KNN module employs position gradients to adaptively establish locality-aware propagation of Identity Encodings, preventing irregular Gaussian spreads near boundaries. We validate the effectiveness of our method through comprehensive experiments. Results show that GradiSeg effectively addresses boundary-related issues, significantly improving segmentation accuracy without compromising scene reconstruction quality. Furthermore, our method's robust segmentation capability and decoupled Identity Encoding representation make it highly suitable for various downstream scene editing tasks, including 3D object removal, swapping and so on.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17421v2">MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 project page: https://www.cis.upenn.edu/~leijh/projects/mosca code release: https://github.com/JiahuiLei/MoSca
    </div>
    <details class="paper-abstract">
      We introduce 4D Motion Scaffolds (MoSca), a modern 4D reconstruction system designed to reconstruct and synthesize novel views of dynamic scenes from monocular videos captured casually in the wild. To address such a challenging and ill-posed inverse problem, we leverage prior knowledge from foundational vision models and lift the video data to a novel Motion Scaffold (MoSca) representation, which compactly and smoothly encodes the underlying motions/deformations. The scene geometry and appearance are then disentangled from the deformation field and are encoded by globally fusing the Gaussians anchored onto the MoSca and optimized via Gaussian Splatting. Additionally, camera focal length and poses can be solved using bundle adjustment without the need of any other pose estimation tools. Experiments demonstrate state-of-the-art performance on dynamic rendering benchmarks and its effectiveness on real videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17660v2">DROID-Splat: Combining end-to-end SLAM with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Recent progress in scene synthesis makes standalone SLAM systems purely based on optimizing hyperprimitives with a Rendering objective possible. However, the tracking performance still lacks behind traditional and end-to-end SLAM systems. An optimal trade-off between robustness, speed and accuracy has not yet been reached, especially for monocular video. In this paper, we introduce a SLAM system based on an end-to-end Tracker and extend it with a Renderer based on recent 3D Gaussian Splatting techniques. Our framework \textbf{DroidSplat} achieves both SotA tracking and rendering results on common SLAM benchmarks. We implemented multiple building blocks of modern SLAM systems to run in parallel, allowing for fast inference on common consumer GPU's. Recent progress in monocular depth prediction and camera calibration allows our system to achieve strong results even on in-the-wild data without known camera intrinsics. Code will be available at \url{https://github.com/ChenHoy/DROID-Splat}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19756v1">DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Gaussian splatting enables fast novel view synthesis in static 3D environments. However, reconstructing real-world environments remains challenging as distractors or occluders break the multi-view consistency assumption required for accurate 3D reconstruction. Most existing methods rely on external semantic information from pre-trained models, introducing additional computational overhead as pre-processing steps or during optimization. In this work, we propose a novel method, DeSplat, that directly separates distractors and static scene elements purely based on volume rendering of Gaussian primitives. We initialize Gaussians within each camera view for reconstructing the view-specific distractors to separately model the static 3D scene and distractors in the alpha compositing stages. DeSplat yields an explicit scene separation of static elements and distractors, achieving comparable results to prior distractor-free approaches without sacrificing rendering speed. We demonstrate DeSplat's effectiveness on three benchmark data sets for distractor-free novel view synthesis. See the project website at https://aaltoml.github.io/desplat/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19654v1">TexGaussian: Generating High-quality PBR Material via Octree-based 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Physically Based Rendering (PBR) materials play a crucial role in modern graphics, enabling photorealistic rendering across diverse environment maps. Developing an effective and efficient algorithm that is capable of automatically generating high-quality PBR materials rather than RGB texture for 3D meshes can significantly streamline the 3D content creation. Most existing methods leverage pre-trained 2D diffusion models for multi-view image synthesis, which often leads to severe inconsistency between the generated textures and input 3D meshes. This paper presents TexGaussian, a novel method that uses octant-aligned 3D Gaussian Splatting for rapid PBR material generation. Specifically, we place each 3D Gaussian on the finest leaf node of the octree built from the input 3D mesh to render the multiview images not only for the albedo map but also for roughness and metallic. Moreover, our model is trained in a regression manner instead of diffusion denoising, capable of generating the PBR material for a 3D mesh in a single feed-forward process. Extensive experiments on publicly available benchmarks demonstrate that our method synthesizes more visually pleasing PBR materials and runs faster than previous methods in both unconditional and text-conditional scenarios, which exhibit better consistency with the given geometry. Our code and trained models are available at https://3d-aigc.github.io/TexGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19594v1">Tortho-Gaussian: Splatting True Digital Orthophoto Maps</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 This work has been submitted to the IEEE Transactions on Geoscience and Remote Sensing for possible publication
    </div>
    <details class="paper-abstract">
      True Digital Orthophoto Maps (TDOMs) are essential products for digital twins and Geographic Information Systems (GIS). Traditionally, TDOM generation involves a complex set of traditional photogrammetric process, which may deteriorate due to various challenges, including inaccurate Digital Surface Model (DSM), degenerated occlusion detections, and visual artifacts in weak texture regions and reflective surfaces, etc. To address these challenges, we introduce TOrtho-Gaussian, a novel method inspired by 3D Gaussian Splatting (3DGS) that generates TDOMs through orthogonal splatting of optimized anisotropic Gaussian kernel. More specifically, we first simplify the orthophoto generation by orthographically splatting the Gaussian kernels onto 2D image planes, formulating a geometrically elegant solution that avoids the need for explicit DSM and occlusion detection. Second, to produce TDOM of large-scale area, a divide-and-conquer strategy is adopted to optimize memory usage and time efficiency of training and rendering for 3DGS. Lastly, we design a fully anisotropic Gaussian kernel that adapts to the varying characteristics of different regions, particularly improving the rendering quality of reflective surfaces and slender structures. Extensive experimental evaluations demonstrate that our method outperforms existing commercial software in several aspects, including the accuracy of building boundaries, the visual quality of low-texture regions and building facades. These results underscore the potential of our approach for large-scale urban scene reconstruction, offering a robust alternative for enhancing TDOM quality and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19588v1">Gaussian Splashing: Direct Volumetric Rendering Underwater</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      In underwater images, most useful features are occluded by water. The extent of the occlusion depends on imaging geometry and can vary even across a sequence of burst images. As a result, 3D reconstruction methods robust on in-air scenes, like Neural Radiance Field methods (NeRFs) or 3D Gaussian Splatting (3DGS), fail on underwater scenes. While a recent underwater adaptation of NeRFs achieved state-of-the-art results, it is impractically slow: reconstruction takes hours and its rendering rate, in frames per second (FPS), is less than 1. Here, we present a new method that takes only a few minutes for reconstruction and renders novel underwater scenes at 140 FPS. Named Gaussian Splashing, our method unifies the strengths and speed of 3DGS with an image formation model for capturing scattering, introducing innovations in the rendering and depth estimation procedures and in the 3DGS loss function. Despite the complexities of underwater adaptation, our method produces images at unparalleled speeds with superior details. Moreover, it reveals distant scene details with far greater clarity than other methods, dramatically improving reconstructed and rendered images. We demonstrate results on existing datasets and a new dataset we have collected. Additional visual results are available at: https://bgu-cs-vil.github.io/gaussiansplashingUW.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19551v1">Bootstraping Clustering of Gaussians for View-consistent 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Injecting semantics into 3D Gaussian Splatting (3DGS) has recently garnered significant attention. While current approaches typically distill 3D semantic features from 2D foundational models (e.g., CLIP and SAM) to facilitate novel view segmentation and semantic understanding, their heavy reliance on 2D supervision can undermine cross-view semantic consistency and necessitate complex data preparation processes, therefore hindering view-consistent scene understanding. In this work, we present FreeGS, an unsupervised semantic-embedded 3DGS framework that achieves view-consistent 3D scene understanding without the need for 2D labels. Instead of directly learning semantic features, we introduce the IDentity-coupled Semantic Field (IDSF) into 3DGS, which captures both semantic representations and view-consistent instance indices for each Gaussian. We optimize IDSF with a two-step alternating strategy: semantics help to extract coherent instances in 3D space, while the resulting instances regularize the injection of stable semantics from 2D space. Additionally, we adopt a 2D-3D joint contrastive loss to enhance the complementarity between view-consistent 3D geometry and rich semantics during the bootstrapping process, enabling FreeGS to uniformly perform tasks such as novel-view semantic segmentation, object selection, and 3D object detection. Extensive experiments on LERF-Mask, 3D-OVS, and ScanNet datasets demonstrate that FreeGS performs comparably to state-of-the-art methods while avoiding the complex data preprocessing workload.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00155v1">T-3DGS: Removing Transient Objects for 3D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      We propose a novel framework to remove transient objects from input videos for 3D scene reconstruction using Gaussian Splatting. Our framework consists of the following steps. In the first step, we propose an unsupervised training strategy for a classification network to distinguish between transient objects and static scene parts based on their different training behavior inside the 3D Gaussian Splatting reconstruction. In the second step, we improve the boundary quality and stability of the detected transients by combining our results from the first step with an off-the-shelf segmentation method. We also propose a simple and effective strategy to track objects in the input video forward and backward in time. Our results show an improvement over the current state of the art in existing sparsely captured datasets and significant improvements in a newly proposed densely captured (video) dataset. More results and code are available at https://transient-3dgs.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19290v1">SADG: Segment Any Dynamic Gaussian Without Object Trackers</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Project page https://yunjinli.github.io/project-sadg
    </div>
    <details class="paper-abstract">
      Understanding dynamic 3D scenes is fundamental for various applications, including extended reality (XR) and autonomous driving. Effectively integrating semantic information into 3D reconstruction enables holistic representation that opens opportunities for immersive and interactive applications. We introduce SADG, Segment Any Dynamic Gaussian Without Object Trackers, a novel approach that combines dynamic Gaussian Splatting representation and semantic information without reliance on object IDs. In contrast to existing works, we do not rely on supervision based on object identities to enable consistent segmentation of dynamic 3D objects. To this end, we propose to learn semantically-aware features by leveraging masks generated from the Segment Anything Model (SAM) and utilizing our novel contrastive learning objective based on hard pixel mining. The learned Gaussian features can be effectively clustered without further post-processing. This enables fast computation for further object-level editing, such as object removal, composition, and style transfer by manipulating the Gaussians in the scene. We further extend several dynamic novel-view datasets with segmentation benchmarks to enable testing of learned feature fields from unseen viewpoints. We evaluate SADG on proposed benchmarks and demonstrate the superior performance of our approach in segmenting objects within dynamic scenes along with its effectiveness for further downstream editing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18929v3">DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 ECCV 2024. Project Page: https://silent-chen.github.io/DGE/
    </div>
    <details class="paper-abstract">
      We consider the problem of editing 3D objects and scenes based on open-ended language instructions. A common approach to this problem is to use a 2D image generator or editor to guide the 3D editing process, obviating the need for 3D data. However, this process is often inefficient due to the need for iterative updates of costly 3D representations, such as neural radiance fields, either through individual view edits or score distillation sampling. A major disadvantage of this approach is the slow convergence caused by aggregating inconsistent information across views, as the guidance from 2D models is not multi-view consistent. We thus introduce the Direct Gaussian Editor (DGE), a method that addresses these issues in two stages. First, we modify a given high-quality image editor like InstructPix2Pix to be multi-view consistent. To do so, we propose a training-free approach that integrates cues from the 3D geometry of the underlying scene. Second, given a multi-view consistent edited sequence of images, we directly and efficiently optimize the 3D representation, which is based on 3D Gaussian Splatting. Because it avoids incremental and iterative edits, DGE is significantly more accurate and efficient than existing approaches and offers additional benefits, such as enabling selective editing of parts of the scene.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00138v1">Unleashing the Power of Data Synthesis in Visual Localization</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 24 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Visual localization, which estimates a camera's pose within a known scene, is a long-standing challenge in vision and robotics. Recent end-to-end methods that directly regress camera poses from query images have gained attention for fast inference. However, existing methods often struggle to generalize to unseen views. In this work, we aim to unleash the power of data synthesis to promote the generalizability of pose regression. Specifically, we lift real 2D images into 3D Gaussian Splats with varying appearance and deblurring abilities, which are then used as a data engine to synthesize more posed images. To fully leverage the synthetic data, we build a two-branch joint training pipeline, with an adversarial discriminator to bridge the syn-to-real gap. Experiments on established benchmarks show that our method outperforms state-of-the-art end-to-end approaches, reducing translation and rotation errors by 50% and 21.6% on indoor datasets, and 35.56% and 38.7% on outdoor datasets. We also validate the effectiveness of our method in dynamic driving scenarios under varying weather conditions. Notably, as data synthesis scales up, our method exhibits a growing ability to interpolate and extrapolate training data for localizing unseen views. Project Page: https://ai4ce.github.io/RAP/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19235v1">InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 technical report, 13 pages
    </div>
    <details class="paper-abstract">
      3D scene understanding has become an essential area of research with applications in autonomous driving, robotics, and augmented reality. Recently, 3D Gaussian Splatting (3DGS) has emerged as a powerful approach, combining explicit modeling with neural adaptability to provide efficient and detailed scene representations. However, three major challenges remain in leveraging 3DGS for scene understanding: 1) an imbalance between appearance and semantics, where dense Gaussian usage for fine-grained texture modeling does not align with the minimal requirements for semantic attributes; 2) inconsistencies between appearance and semantics, as purely appearance-based Gaussians often misrepresent object boundaries; and 3) reliance on top-down instance segmentation methods, which struggle with uneven category distributions, leading to over- or under-segmentation. In this work, we propose InstanceGaussian, a method that jointly learns appearance and semantic features while adaptively aggregating instances. Our contributions include: i) a novel Semantic-Scaffold-GS representation balancing appearance and semantics to improve feature representations and boundary delineation; ii) a progressive appearance-semantic joint training strategy to enhance stability and segmentation accuracy; and iii) a bottom-up, category-agnostic instance aggregation approach that addresses segmentation challenges through farthest point sampling and connected component analysis. Our approach achieves state-of-the-art performance in category-agnostic, open-vocabulary 3D point-level segmentation, highlighting the effectiveness of the proposed representation and training strategies. Project page: https://lhj-git.github.io/InstanceGaussian/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19233v1">Gaussians-to-Life: Text-Driven Animation of 3D Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Project website: https://wimmerth.github.io/gaussians2life.html
    </div>
    <details class="paper-abstract">
      State-of-the-art novel view synthesis methods achieve impressive results for multi-view captures of static 3D scenes. However, the reconstructed scenes still lack "liveliness," a key component for creating engaging 3D experiences. Recently, novel video diffusion models generate realistic videos with complex motion and enable animations of 2D images, however they cannot naively be used to animate 3D scenes as they lack multi-view consistency. To breathe life into the static world, we propose Gaussians2Life, a method for animating parts of high-quality 3D scenes in a Gaussian Splatting representation. Our key idea is to leverage powerful video diffusion models as the generative component of our model and to combine these with a robust technique to lift 2D videos into meaningful 3D motion. We find that, in contrast to prior work, this enables realistic animations of complex, pre-existing 3D scenes and further enables the animation of a large variety of object classes, while related work is mostly focused on prior-based character animation, or single 3D objects. Our model enables the creation of consistent, immersive 3D experiences for arbitrary scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11868v9">View-Consistent 3D Editing with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 accepted to ECCV 2024
    </div>
    <details class="paper-abstract">
      The advent of 3D Gaussian Splatting (3DGS) has revolutionized 3D editing, offering efficient, high-fidelity rendering and enabling precise local manipulations. Currently, diffusion-based 2D editing models are harnessed to modify multi-view rendered images, which then guide the editing of 3DGS models. However, this approach faces a critical issue of multi-view inconsistency, where the guidance images exhibit significant discrepancies across views, leading to mode collapse and visual artifacts of 3DGS. To this end, we introduce View-consistent Editing (VcEdit), a novel framework that seamlessly incorporates 3DGS into image editing processes, ensuring multi-view consistency in edited guidance images and effectively mitigating mode collapse issues. VcEdit employs two innovative consistency modules: the Cross-attention Consistency Module and the Editing Consistency Module, both designed to reduce inconsistencies in edited images. By incorporating these consistency modules into an iterative pattern, VcEdit proficiently resolves the issue of multi-view inconsistency, facilitating high-quality 3DGS editing across a diverse range of scenes. Further video results are shown in http://vcedit.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18966v1">SuperGaussians: Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      Gaussian Splattings demonstrate impressive results in multi-view reconstruction based on Gaussian explicit representations. However, the current Gaussian primitives only have a single view-dependent color and an opacity to represent the appearance and geometry of the scene, resulting in a non-compact representation. In this paper, we introduce a new method called SuperGaussians that utilizes spatially varying colors and opacity in a single Gaussian primitive to improve its representation ability. We have implemented bilinear interpolation, movable kernels, and even tiny neural networks as spatially varying functions. Quantitative and qualitative experimental results demonstrate that all three functions outperform the baseline, with the best movable kernels achieving superior novel view synthesis performance on multiple datasets, highlighting the strong potential of spatially varying functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17190v3">SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Project page: https://gynjn.github.io/selfsplat/
    </div>
    <details class="paper-abstract">
      We propose SelfSplat, a novel 3D Gaussian Splatting model designed to perform pose-free and 3D prior-free generalizable 3D reconstruction from unposed multi-view images. These settings are inherently ill-posed due to the lack of ground-truth data, learned geometric information, and the need to achieve accurate 3D reconstruction without finetuning, making it difficult for conventional methods to achieve high-quality results. Our model addresses these challenges by effectively integrating explicit 3D representations with self-supervised depth and pose estimation techniques, resulting in reciprocal improvements in both pose accuracy and 3D reconstruction quality. Furthermore, we incorporate a matching-aware pose estimation network and a depth refinement module to enhance geometry consistency across views, ensuring more accurate and stable 3D reconstructions. To present the performance of our method, we evaluated it on large-scale real-world datasets, including RealEstate10K, ACID, and DL3DV. SelfSplat achieves superior results over previous state-of-the-art methods in both appearance and geometry quality, also demonstrates strong cross-dataset generalization capabilities. Extensive ablation studies and analysis also validate the effectiveness of our proposed methods. Code and pretrained models are available at https://gynjn.github.io/selfsplat/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14514v4">NexusSplats: Efficient 3D Gaussian Splatting in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Project page: https://nexus-splats.github.io/
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3DGS) has recently demonstrated remarkable rendering quality and efficiency in 3D scene reconstruction, it struggles with varying lighting conditions and incidental occlusions in real-world scenarios. To accommodate varying lighting conditions, existing 3DGS extensions apply color mapping to the massive Gaussian primitives with individually optimized appearance embeddings. To handle occlusions, they predict pixel-wise uncertainties via 2D image features for occlusion capture. Nevertheless, such massive color mapping and pixel-wise uncertainty prediction strategies suffer from not only additional computational costs but also coarse-grained lighting and occlusion handling. In this work, we propose a nexus kernel-driven approach, termed NexusSplats, for efficient and finer 3D scene reconstruction under complex lighting and occlusion conditions. In particular, NexusSplats leverages a novel light decoupling strategy where appearance embeddings are optimized based on nexus kernels instead of massive Gaussian primitives, thus accelerating reconstruction speeds while ensuring local color consistency for finer textures. Additionally, a Gaussian-wise uncertainty mechanism is developed, aligning 3D structures with 2D image features for fine-grained occlusion handling. Experimental results demonstrate that NexusSplats achieves state-of-the-art rendering quality while reducing reconstruction time by up to 70.4% compared to the current best in quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18866v1">RIGI: Rectifying Image-to-3D Generation Inconsistency via Uncertainty-aware Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Project Page: https://rigi3d.github.io/
    </div>
    <details class="paper-abstract">
      Given a single image of a target object, image-to-3D generation aims to reconstruct its texture and geometric shape. Recent methods often utilize intermediate media, such as multi-view images or videos, to bridge the gap between input image and the 3D target, thereby guiding the generation of both shape and texture. However, inconsistencies in the generated multi-view snapshots frequently introduce noise and artifacts along object boundaries, undermining the 3D reconstruction process. To address this challenge, we leverage 3D Gaussian Splatting (3DGS) for 3D reconstruction, and explicitly integrate uncertainty-aware learning into the reconstruction process. By capturing the stochasticity between two Gaussian models, we estimate an uncertainty map, which is subsequently used for uncertainty-aware regularization to rectify the impact of inconsistencies. Specifically, we optimize both Gaussian models simultaneously, calculating the uncertainty map by evaluating the discrepancies between rendered images from identical viewpoints. Based on the uncertainty map, we apply adaptive pixel-wise loss weighting to regularize the models, reducing reconstruction intensity in high-uncertainty regions. This approach dynamically detects and mitigates conflicts in multi-view labels, leading to smoother results and effectively reducing artifacts. Extensive experiments show the effectiveness of our method in improving 3D generation quality by reducing inconsistencies and artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.18072v1">SmileSplat: Generalizable Gaussian Splats for Unconstrained Sparse Images</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      Sparse Multi-view Images can be Learned to predict explicit radiance fields via Generalizable Gaussian Splatting approaches, which can achieve wider application prospects in real-life when ground-truth camera parameters are not required as inputs. In this paper, a novel generalizable Gaussian Splatting method, SmileSplat, is proposed to reconstruct pixel-aligned Gaussian surfels for diverse scenarios only requiring unconstrained sparse multi-view images. First, Gaussian surfels are predicted based on the multi-head Gaussian regression decoder, which can are represented with less degree-of-freedom but have better multi-view consistency. Furthermore, the normal vectors of Gaussian surfel are enhanced based on high-quality of normal priors. Second, the Gaussians and camera parameters (both extrinsic and intrinsic) are optimized to obtain high-quality Gaussian radiance fields for novel view synthesis tasks based on the proposed Bundle-Adjusting Gaussian Splatting module. Extensive experiments on novel view rendering and depth map prediction tasks are conducted on public datasets, demonstrating that the proposed method achieves state-of-the-art performance in various 3D vision tasks. More information can be found on our project page (https://yanyan-li.github.io/project/gs/smilesplat)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18625v1">Textured Gaussians for Enhanced 3D Scene Appearance Modeling</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Project website: https://textured-gaussians.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as a state-of-the-art 3D reconstruction and rendering technique due to its high-quality results and fast training and rendering time. However, pixels covered by the same Gaussian are always shaded in the same color up to a Gaussian falloff scaling factor. Furthermore, the finest geometric detail any individual Gaussian can represent is a simple ellipsoid. These properties of 3DGS greatly limit the expressivity of individual Gaussian primitives. To address these issues, we draw inspiration from texture and alpha mapping in traditional graphics and integrate it with 3DGS. Specifically, we propose a new generalized Gaussian appearance representation that augments each Gaussian with alpha~(A), RGB, or RGBA texture maps to model spatially varying color and opacity across the extent of each Gaussian. As such, each Gaussian can represent a richer set of texture patterns and geometric structures, instead of just a single color and ellipsoid as in naive Gaussian Splatting. Surprisingly, we found that the expressivity of Gaussians can be greatly improved by using alpha-only texture maps, and further augmenting Gaussians with RGB texture maps achieves the highest expressivity. We validate our method on a wide variety of standard benchmark datasets and our own custom captures at both the object and scene levels. We demonstrate image quality improvements over existing methods while using a similar or lower number of Gaussians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18675v1">GaussianSpeech: Audio-Driven Gaussian Avatars</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Paper Video: https://youtu.be/2VqYoFlYcwQ Project Page: https://shivangi-aneja.github.io/projects/gaussianspeech
    </div>
    <details class="paper-abstract">
      We introduce GaussianSpeech, a novel approach that synthesizes high-fidelity animation sequences of photo-realistic, personalized 3D human head avatars from spoken audio. To capture the expressive, detailed nature of human heads, including skin furrowing and finer-scale facial movements, we propose to couple speech signal with 3D Gaussian splatting to create realistic, temporally coherent motion sequences. We propose a compact and efficient 3DGS-based avatar representation that generates expression-dependent color and leverages wrinkle- and perceptually-based losses to synthesize facial details, including wrinkles that occur with different expressions. To enable sequence modeling of 3D Gaussian splats with audio, we devise an audio-conditioned transformer model capable of extracting lip and expression features directly from audio input. Due to the absence of high-quality datasets of talking humans in correspondence with audio, we captured a new large-scale multi-view dataset of audio-visual sequences of talking humans with native English accents and diverse facial geometry. GaussianSpeech consistently achieves state-of-the-art performance with visually natural motion at real time rendering rates, while encompassing diverse facial expressions and styles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18548v1">PhyCAGE: Physically Plausible Compositional 3D Asset Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Project page: https://wolfball.github.io/phycage/
    </div>
    <details class="paper-abstract">
      We present PhyCAGE, the first approach for physically plausible compositional 3D asset generation from a single image. Given an input image, we first generate consistent multi-view images for components of the assets. These images are then fitted with 3D Gaussian Splatting representations. To ensure that the Gaussians representing objects are physically compatible with each other, we introduce a Physical Simulation-Enhanced Score Distillation Sampling (PSE-SDS) technique to further optimize the positions of the Gaussians. It is achieved by setting the gradient of the SDS loss as the initial velocity of the physical simulation, allowing the simulator to act as a physics-guided optimizer that progressively corrects the Gaussians' positions to a physically compatible state. Experimental results demonstrate that the proposed method can generate physically plausible compositional 3D assets given a single image.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18667v1">Point Cloud Unsupervised Pre-training via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 14 pages, 4 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Pre-training on large-scale unlabeled datasets contribute to the model achieving powerful performance on 3D vision tasks, especially when annotations are limited. However, existing rendering-based self-supervised frameworks are computationally demanding and memory-intensive during pre-training due to the inherent nature of volume rendering. In this paper, we propose an efficient framework named GS$^3$ to learn point cloud representation, which seamlessly integrates fast 3D Gaussian Splatting into the rendering-based framework. The core idea behind our framework is to pre-train the point cloud encoder by comparing rendered RGB images with real RGB images, as only Gaussian points enriched with learned rich geometric and appearance information can produce high-quality renderings. Specifically, we back-project the input RGB-D images into 3D space and use a point cloud encoder to extract point-wise features. Then, we predict 3D Gaussian points of the scene from the learned point cloud features and uses a tile-based rasterizer for image rendering. Finally, the pre-trained point cloud encoder can be fine-tuned to adapt to various downstream 3D tasks, including high-level perception tasks such as 3D segmentation and detection, as well as low-level tasks such as 3D scene reconstruction. Extensive experiments on downstream tasks demonstrate the strong transferability of the pre-trained point cloud encoder and the effectiveness of our self-supervised learning framework. In addition, our GS$^3$ framework is highly efficient, achieving approximately 9$\times$ pre-training speedup and less than 0.25$\times$ memory cost compared to the previous rendering-based framework Ponder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18473v1">HEMGS: A Hybrid Entropy Model for 3D Gaussian Splatting Data Compression</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Fast progress in 3D Gaussian Splatting (3DGS) has made 3D Gaussians popular for 3D modeling and image rendering, but this creates big challenges in data storage and transmission. To obtain a highly compact 3DGS representation, we propose a hybrid entropy model for Gaussian Splatting (HEMGS) data compression, which comprises two primary components, a hyperprior network and an autoregressive network. To effectively reduce structural redundancy across attributes, we apply a progressive coding algorithm to generate hyperprior features, in which we use previously compressed attributes and location as prior information. In particular, to better extract the location features from these compressed attributes, we adopt a domain-aware and instance-aware architecture to respectively capture domain-aware structural relations without additional storage costs and reveal scene-specific features through MLPs. Additionally, to reduce redundancy within each attribute, we leverage relationships between neighboring compressed elements within the attributes through an autoregressive network. Given its unique structure, we propose an adaptive context coding algorithm with flexible receptive fields to effectively capture adjacent compressed elements. Overall, we integrate our HEMGS into an end-to-end optimized 3DGS compression framework and the extensive experimental results on four benchmarks indicate that our method achieves about 40\% average reduction in size while maintaining the rendering quality over our baseline method and achieving state-of-the-art compression results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18311v1">Neural Surface Priors for Editable Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      In computer graphics, there is a need to recover easily modifiable representations of 3D geometry and appearance from image data. We introduce a novel method for this task using 3D Gaussian Splatting, which enables intuitive scene editing through mesh adjustments. Starting with input images and camera poses, we reconstruct the underlying geometry using a neural Signed Distance Field and extract a high-quality mesh. Our model then estimates a set of Gaussians, where each component is flat, and the opacity is conditioned on the recovered neural surface. To facilitate editing, we produce a proxy representation that encodes information about the Gaussians' shape and position. Unlike other methods, our pipeline allows modifications applied to the extracted mesh to be propagated to the proxy representation, from which we recover the updated parameters of the Gaussians. This effectively transfers the mesh edits back to the recovered appearance representation. By leveraging mesh-guided transformations, our approach simplifies 3D scene editing and offers improvements over existing methods in terms of usability and visual fidelity of edits. The complete source code for this project can be accessed at \url{https://github.com/WJakubowska/NeuralSurfacePriors}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16816v2">SplatAD: Real-Time Lidar and Camera Rendering with 3D Gaussian Splatting for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous robots, such as self-driving vehicles, requires extensive testing across diverse driving scenarios. Simulation is a key ingredient for conducting such testing in a cost-effective and scalable way. Neural rendering methods have gained popularity, as they can build simulation environments from collected logs in a data-driven manner. However, existing neural radiance field (NeRF) methods for sensor-realistic rendering of camera and lidar data suffer from low rendering speeds, limiting their applicability for large-scale testing. While 3D Gaussian Splatting (3DGS) enables real-time rendering, current methods are limited to camera data and are unable to render lidar data essential for autonomous driving. To address these limitations, we propose SplatAD, the first 3DGS-based method for realistic, real-time rendering of dynamic scenes for both camera and lidar data. SplatAD accurately models key sensor-specific phenomena such as rolling shutter effects, lidar intensity, and lidar ray dropouts, using purpose-built algorithms to optimize rendering efficiency. Evaluation across three autonomous driving datasets demonstrates that SplatAD achieves state-of-the-art rendering quality with up to +2 PSNR for NVS and +3 PSNR for reconstruction while increasing rendering speed over NeRF-based methods by an order of magnitude. See https://research.zenseact.com/publications/splatad/ for our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18072v1">SmileSplat: Generalizable Gaussian Splats for Unconstrained Sparse Images</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Sparse Multi-view Images can be Learned to predict explicit radiance fields via Generalizable Gaussian Splatting approaches, which can achieve wider application prospects in real-life when ground-truth camera parameters are not required as inputs. In this paper, a novel generalizable Gaussian Splatting method, SmileSplat, is proposed to reconstruct pixel-aligned Gaussian surfels for diverse scenarios only requiring unconstrained sparse multi-view images. First, Gaussian surfels are predicted based on the multi-head Gaussian regression decoder, which can are represented with less degree-of-freedom but have better multi-view consistency. Furthermore, the normal vectors of Gaussian surfel are enhanced based on high-quality of normal priors. Second, the Gaussians and camera parameters (both extrinsic and intrinsic) are optimized to obtain high-quality Gaussian radiance fields for novel view synthesis tasks based on the proposed Bundle-Adjusting Gaussian Splatting module. Extensive experiments on novel view rendering and depth map prediction tasks are conducted on public datasets, demonstrating that the proposed method achieves state-of-the-art performance in various 3D vision tasks. More information can be found on our project page (https://yanyan-li.github.io/project/gs/smilesplat)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18066v1">GLS: Geometry-aware 3D Language Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has achieved significant performance on indoor surface reconstruction and open-vocabulary segmentation. This paper presents GLS, a unified framework of surface reconstruction and open-vocabulary segmentation based on 3DGS. GLS extends two fields by exploring the correlation between them. For indoor surface reconstruction, we introduce surface normal prior as a geometric cue to guide the rendered normal, and use the normal error to optimize the rendered depth. For open-vocabulary segmentation, we employ 2D CLIP features to guide instance features and utilize DEVA masks to enhance their view consistency. Extensive experiments demonstrate the effectiveness of jointly optimizing surface reconstruction and open-vocabulary segmentation, where GLS surpasses state-of-the-art approaches of each task on MuSHRoom, ScanNet++, and LERF-OVS datasets. Code will be available at https://github.com/JiaxiongQ/GLS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17982v1">HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Under review process
    </div>
    <details class="paper-abstract">
      We present HI-SLAM2, a geometry-aware Gaussian SLAM system that achieves fast and accurate monocular scene reconstruction using only RGB input. Existing Neural SLAM or 3DGS-based SLAM methods often trade off between rendering quality and geometry accuracy, our research demonstrates that both can be achieved simultaneously with RGB input alone. The key idea of our approach is to enhance the ability for geometry estimation by combining easy-to-obtain monocular priors with learning-based dense SLAM, and then using 3D Gaussian splatting as our core map representation to efficiently model the scene. Upon loop closure, our method ensures on-the-fly global consistency through efficient pose graph bundle adjustment and instant map updates by explicitly deforming the 3D Gaussian units based on anchored keyframe updates. Furthermore, we introduce a grid-based scale alignment strategy to maintain improved scale consistency in prior depths for finer depth details. Through extensive experiments on Replica, ScanNet, and ScanNet++, we demonstrate significant improvements over existing Neural SLAM methods and even surpass RGB-D-based methods in both reconstruction and rendering quality. The project page and source code will be made available at https://hi-slam2.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2311.11221v3">GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Text-to-3D, known for its efficient generation methods and expansive creative potential, has garnered significant attention in the AIGC domain. However, the pixel-wise rendering of NeRF and its ray marching light sampling constrain the rendering speed, impacting its utility in downstream industrial applications. Gaussian Splatting has recently shown a trend of replacing the traditional pointwise sampling technique commonly used in NeRF-based methodologies, and it is changing various aspects of 3D reconstruction. This paper introduces a novel text to 3D content generation framework, Gaussian Diffusion, based on Gaussian Splatting and produces more realistic renderings. The challenge of achieving multi-view consistency in 3D generation significantly impedes modeling complexity and accuracy. Taking inspiration from SJC, we explore employing multi-view noise distributions to perturb images generated by 3D Gaussian Splatting, aiming to rectify inconsistencies in multi-view geometry. We ingeniously devise an efficient method to generate noise that produces Gaussian noise from diverse viewpoints, all originating from a shared noise source. Furthermore, vanilla 3D Gaussian-based generation tends to trap models in local minima, causing artifacts like floaters, burrs, or proliferative elements. To mitigate these issues, we propose the variational Gaussian Splatting technique to enhance the quality and stability of 3D appearance. To our knowledge, our approach represents the first comprehensive utilization of Gaussian Diffusion across the entire spectrum of 3D content generation processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17605v1">Distractor-free Generalizable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      We present DGGS, a novel framework addressing the previously unexplored challenge of Distractor-free Generalizable 3D Gaussian Splatting (3DGS). It accomplishes two key objectives: fortifying generalizable 3DGS against distractor-laden data during both training and inference phases, while successfully extending cross-scene adaptation capabilities to conventional distractor-free approaches. To achieve these objectives, DGGS introduces a scene-agnostic reference-based mask prediction and refinement methodology during training phase, coupled with a training view selection strategy, effectively improving distractor prediction accuracy and training stability. Moreover, to address distractor-induced voids and artifacts during inference stage, we propose a two-stage inference framework for better reference selection based on the predicted distractor masks, complemented by a distractor pruning module to eliminate residual distractor effects. Extensive generalization experiments demonstrate DGGS's advantages under distractor-laden conditions. Additionally, experimental results show that our scene-agnostic mask inference achieves accuracy comparable to scene-specific trained methods. Homepage is \url{https://github.com/bbbbby-99/DGGS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11221v3">GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Text-to-3D, known for its efficient generation methods and expansive creative potential, has garnered significant attention in the AIGC domain. However, the pixel-wise rendering of NeRF and its ray marching light sampling constrain the rendering speed, impacting its utility in downstream industrial applications. Gaussian Splatting has recently shown a trend of replacing the traditional pointwise sampling technique commonly used in NeRF-based methodologies, and it is changing various aspects of 3D reconstruction. This paper introduces a novel text to 3D content generation framework, Gaussian Diffusion, based on Gaussian Splatting and produces more realistic renderings. The challenge of achieving multi-view consistency in 3D generation significantly impedes modeling complexity and accuracy. Taking inspiration from SJC, we explore employing multi-view noise distributions to perturb images generated by 3D Gaussian Splatting, aiming to rectify inconsistencies in multi-view geometry. We ingeniously devise an efficient method to generate noise that produces Gaussian noise from diverse viewpoints, all originating from a shared noise source. Furthermore, vanilla 3D Gaussian-based generation tends to trap models in local minima, causing artifacts like floaters, burrs, or proliferative elements. To mitigate these issues, we propose the variational Gaussian Splatting technique to enhance the quality and stability of 3D appearance. To our knowledge, our approach represents the first comprehensive utilization of Gaussian Diffusion across the entire spectrum of 3D content generation processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19702v4">RNG: Relightable Neural Gaussians</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 Submission version
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown impressive results for the novel view synthesis task, where lighting is assumed to be fixed. However, creating relightable 3D assets, especially for objects with ill-defined shapes (fur, fabric, etc.), remains a challenging task. The decomposition between light, geometry, and material is ambiguous, especially if either smooth surface assumptions or surfacebased analytical shading models do not apply. We propose Relightable Neural Gaussians (RNG), a novel 3DGS-based framework that enables the relighting of objects with both hard surfaces or soft boundaries, while avoiding assumptions on the shading model. We condition the radiance at each point on both view and light directions. We also introduce a shadow cue, as well as a depth refinement network to improve shadow accuracy. Finally, we propose a hybrid forward-deferred fitting strategy to balance geometry and appearance quality. Our method achieves significantly faster training (1.3 hours) and rendering (60 frames per second) compared to a prior method based on neural radiance fields and produces higher-quality shadows than a concurrent 3DGS-based method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14974v2">3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 13 pages, 13 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Recent advances in radiance field reconstruction, such as 3D Gaussian Splatting (3DGS), have achieved high-quality novel view synthesis and fast rendering by representing scenes with compositions of Gaussian primitives. However, 3D Gaussians present several limitations for scene reconstruction. Accurately capturing hard edges is challenging without significantly increasing the number of Gaussians, creating a large memory footprint. Moreover, they struggle to represent flat surfaces, as they are diffused in space. Without hand-crafted regularizers, they tend to disperse irregularly around the actual surface. To circumvent these issues, we introduce a novel method, named 3D Convex Splatting (3DCS), which leverages 3D smooth convexes as primitives for modeling geometrically-meaningful radiance fields from multi-view images. Smooth convex shapes offer greater flexibility than Gaussians, allowing for a better representation of 3D scenes with hard edges and dense volumes using fewer primitives. Powered by our efficient CUDA-based rasterizer, 3DCS achieves superior performance over 3DGS on benchmarks such as Mip-NeRF360, Tanks and Temples, and Deep Blending. Specifically, our method attains an improvement of up to 0.81 in PSNR and 0.026 in LPIPS compared to 3DGS while maintaining high rendering speeds and reducing the number of required primitives. Our results highlight the potential of 3D Convex Splatting to become the new standard for high-quality scene reconstruction and novel view synthesis. Project page: convexsplatting.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14384v2">Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 A novel one-stage 3DGS-based diffusion generates objects and scenes from a single view in ~6 seconds
    </div>
    <details class="paper-abstract">
      Existing feed-forward image-to-3D methods mainly rely on 2D multi-view diffusion models that cannot guarantee 3D consistency. These methods easily collapse when changing the prompt view direction and mainly handle object-centric prompt images. In this paper, we propose a novel single-stage 3D diffusion model, DiffusionGS, for object and scene generation from a single view. DiffusionGS directly outputs 3D Gaussian point clouds at each timestep to enforce view consistency and allow the model to generate robustly given prompt views of any directions, beyond object-centric inputs. Plus, to improve the capability and generalization ability of DiffusionGS, we scale up 3D training data by developing a scene-object mixed training strategy. Experiments show that our method enjoys better generation quality (2.20 dB higher in PSNR and 23.25 lower in FID) and over 5x faster speed (~6s on an A100 GPU) than SOTA methods. The user study and text-to-3D applications also reveals the practical values of our method. Our Project page at https://caiyuanhao1998.github.io/project/DiffusionGS/ shows the video and interactive generation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17044v1">4D Scaffold Gaussian Splatting for Memory Efficient Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Existing 4D Gaussian methods for dynamic scene reconstruction offer high visual fidelity and fast rendering. However, these methods suffer from excessive memory and storage demands, which limits their practical deployment. This paper proposes a 4D anchor-based framework that retains visual quality and rendering speed of 4D Gaussians while significantly reducing storage costs. Our method extends 3D scaffolding to 4D space, and leverages sparse 4D grid-aligned anchors with compressed feature vectors. Each anchor models a set of neural 4D Gaussians, each of which represent a local spatiotemporal region. In addition, we introduce a temporal coverage-aware anchor growing strategy to effectively assign additional anchors to under-reconstructed dynamic regions. Our method adjusts the accumulated gradients based on Gaussians' temporal coverage, improving reconstruction quality in dynamic regions. To reduce the number of anchors, we further present enhanced formulations of neural 4D Gaussians. These include the neural velocity, and the temporal opacity derived from a generalized Gaussian distribution. Experimental results demonstrate that our method achieves state-of-the-art visual quality and 97.8% storage reduction over 4DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.15476v1">Gassidy: Gaussian Splatting SLAM in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 This paper is currently under reviewed for ICRA 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) allows flexible adjustments to scene representation, enabling continuous optimization of scene quality during dense visual simultaneous localization and mapping (SLAM) in static environments. However, 3DGS faces challenges in handling environmental disturbances from dynamic objects with irregular movement, leading to degradation in both camera tracking accuracy and map reconstruction quality. To address this challenge, we develop an RGB-D dense SLAM which is called Gaussian Splatting SLAM in Dynamic Environments (Gassidy). This approach calculates Gaussians to generate rendering loss flows for each environmental component based on a designed photometric-geometric loss function. To distinguish and filter environmental disturbances, we iteratively analyze rendering loss flows to detect features characterized by changes in loss values between dynamic objects and static components. This process ensures a clean environment for accurate scene reconstruction. Compared to state-of-the-art SLAM methods, experimental results on open datasets show that Gassidy improves camera tracking precision by up to 97.9% and enhances map quality by up to 6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16898v1">G2SDF: Surface Reconstruction from Explicit Gaussians with Implicit SDFs</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      State-of-the-art novel view synthesis methods such as 3D Gaussian Splatting (3DGS) achieve remarkable visual quality. While 3DGS and its variants can be rendered efficiently using rasterization, many tasks require access to the underlying 3D surface, which remains challenging to extract due to the sparse and explicit nature of this representation. In this paper, we introduce G2SDF, a novel approach that addresses this limitation by integrating a neural implicit Signed Distance Field (SDF) into the Gaussian Splatting framework. Our method links the opacity values of Gaussians with their distances to the surface, ensuring a closer alignment of Gaussians with the scene surface. To extend this approach to unbounded scenes at varying scales, we propose a normalization function that maps any range to a fixed interval. To further enhance reconstruction quality, we leverage an off-the-shelf depth estimator as pseudo ground truth during Gaussian Splatting optimization. By establishing a differentiable connection between the explicit Gaussians and the implicit SDF, our approach enables high-quality surface reconstruction and rendering. Experimental results on several real-world datasets demonstrate that G2SDF achieves superior reconstruction quality than prior works while maintaining the efficiency of 3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16877v1">PreF3R: Pose-Free Feed-Forward 3D Gaussian Splatting from Variable-length Image Sequence</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 project page: https://computationalrobotics.seas.harvard.edu/PreF3R/
    </div>
    <details class="paper-abstract">
      We present PreF3R, Pose-Free Feed-forward 3D Reconstruction from an image sequence of variable length. Unlike previous approaches, PreF3R removes the need for camera calibration and reconstructs the 3D Gaussian field within a canonical coordinate frame directly from a sequence of unposed images, enabling efficient novel-view rendering. We leverage DUSt3R's ability for pair-wise 3D structure reconstruction, and extend it to sequential multi-view input via a spatial memory network, eliminating the need for optimization-based global alignment. Additionally, PreF3R incorporates a dense Gaussian parameter prediction head, which enables subsequent novel-view synthesis with differentiable rasterization. This allows supervising our model with the combination of photometric loss and pointmap regression loss, enhancing both photorealism and structural accuracy. Given a sequence of ordered images, PreF3R incrementally reconstructs the 3D Gaussian field at 20 FPS, therefore enabling real-time novel-view rendering. Empirical experiments demonstrate that PreF3R is an effective solution for the challenging task of pose-free feed-forward novel-view synthesis, while also exhibiting robust generalization to unseen scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16443v1">SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 Project Page: https://gohyojun15.github.io/SplatFlow/
    </div>
    <details class="paper-abstract">
      Text-based generation and editing of 3D scenes hold significant potential for streamlining content creation through intuitive user interactions. While recent advances leverage 3D Gaussian Splatting (3DGS) for high-fidelity and real-time rendering, existing methods are often specialized and task-focused, lacking a unified framework for both generation and editing. In this paper, we introduce SplatFlow, a comprehensive framework that addresses this gap by enabling direct 3DGS generation and editing. SplatFlow comprises two main components: a multi-view rectified flow (RF) model and a Gaussian Splatting Decoder (GSDecoder). The multi-view RF model operates in latent space, generating multi-view images, depths, and camera poses simultaneously, conditioned on text prompts, thus addressing challenges like diverse scene scales and complex camera trajectories in real-world settings. Then, the GSDecoder efficiently translates these latent outputs into 3DGS representations through a feed-forward 3DGS method. Leveraging training-free inversion and inpainting techniques, SplatFlow enables seamless 3DGS editing and supports a broad range of 3D tasks-including object editing, novel view synthesis, and camera pose estimation-within a unified framework without requiring additional complex pipelines. We validate SplatFlow's capabilities on the MVImgNet and DL3DV-7K datasets, demonstrating its versatility and effectiveness in various 3D generation, editing, and inpainting-based tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16392v1">Quadratic Gaussian Splatting for Efficient and Detailed Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has attracted attention for its superior rendering quality and speed over Neural Radiance Fields (NeRF). To address 3DGS's limitations in surface representation, 2D Gaussian Splatting (2DGS) introduced disks as scene primitives to model and reconstruct geometries from multi-view images, offering view-consistent geometry. However, the disk's first-order linear approximation often leads to over-smoothed results. We propose Quadratic Gaussian Splatting (QGS), a novel method that replaces disks with quadric surfaces, enhancing geometric fitting, whose code will be open-sourced. QGS defines Gaussian distributions in non-Euclidean space, allowing primitives to capture more complex textures. As a second-order surface approximation, QGS also renders spatial curvature to guide the normal consistency term, to effectively reduce over-smoothing. Moreover, QGS is a generalized version of 2DGS that achieves more accurate and detailed reconstructions, as verified by experiments on DTU and TNT, demonstrating its effectiveness in surpassing current state-of-the-art methods in geometry reconstruction. Our code willbe released as open source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18394v2">Reconstructing Satellites in 3D from Amateur Telescope Images</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      This paper proposes a framework for the 3D reconstruction of satellites in low-Earth orbit, utilizing videos captured by small amateur telescopes. The video data obtained from these telescopes differ significantly from data for standard 3D reconstruction tasks, characterized by intense motion blur, atmospheric turbulence, pervasive background light pollution, extended focal length and constrained observational perspectives. To address these challenges, our approach begins with a comprehensive pre-processing workflow that encompasses deep learning-based image restoration, feature point extraction and camera pose initialization. We apply a customized Structure from Motion (SfM) approach, followed by an improved 3D Gaussian splatting algorithm, to achieve high-fidelity 3D model reconstruction. Our technique supports simultaneous 3D Gaussian training and pose estimation, enabling the robust generation of intricate 3D point clouds from sparse, noisy data. The procedure is further bolstered by a post-editing phase designed to eliminate noise points inconsistent with our prior knowledge of a satellite's geometric constraints. We validate our approach on synthetic datasets and actual observations of China's Space Station and International Space Station, showcasing its significant advantages over existing methods in reconstructing 3D space objects from ground-based observations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16180v1">Event-boosted Deformable 3D Gaussians for Fast Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) enables real-time rendering but struggles with fast motion due to low temporal resolution of RGB cameras. To address this, we introduce the first approach combining event cameras, which capture high-temporal-resolution, continuous motion data, with deformable 3D-GS for fast dynamic scene reconstruction. We observe that threshold modeling for events plays a crucial role in achieving high-quality reconstruction. Therefore, we propose a GS-Threshold Joint Modeling (GTJM) strategy, creating a mutually reinforcing process that greatly improves both 3D reconstruction and threshold modeling. Moreover, we introduce a Dynamic-Static Decomposition (DSD) strategy that first identifies dynamic areas by exploiting the inability of static Gaussians to represent motions, then applies a buffer-based soft decomposition to separate dynamic and static areas. This strategy accelerates rendering by avoiding unnecessary deformation in static areas, and focuses on dynamic areas to enhance fidelity. Our approach achieves high-fidelity dynamic reconstruction at 156 FPS with a 400$\times$400 resolution on an RTX 3090 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16779v1">NovelGS: Consistent Novel-view Denoising via Large Gaussian Reconstruction Model</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      We introduce NovelGS, a diffusion model for Gaussian Splatting (GS) given sparse-view images. Recent works leverage feed-forward networks to generate pixel-aligned Gaussians, which could be fast rendered. Unfortunately, the method was unable to produce satisfactory results for areas not covered by the input images due to the formulation of these methods. In contrast, we leverage the novel view denoising through a transformer-based network to generate 3D Gaussians. Specifically, by incorporating both conditional views and noisy target views, the network predicts pixel-aligned Gaussians for each view. During training, the rendered target and some additional views of the Gaussians are supervised. During inference, the target views are iteratively rendered and denoised from pure noise. Our approach demonstrates state-of-the-art performance in addressing the multi-view image reconstruction challenge. Due to generative modeling of unseen regions, NovelGS effectively reconstructs 3D objects with consistent and sharp textures. Experimental results on publicly available datasets indicate that NovelGS substantially surpasses existing image-to-3D frameworks, both qualitatively and quantitatively. We also demonstrate the potential of NovelGS in generative tasks, such as text-to-3D and image-to-3D, by integrating it with existing multiview diffusion models. We will make the code publicly accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09156v2">DyGASR: Dynamic Generalized Exponential Splatting with Surface Alignment for Accelerated 3D Mesh Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D Gaussian Splatting (3DGS), which lead to high-quality novel view synthesis and accelerated rendering, have remarkably improved the quality of radiance field reconstruction. However, the extraction of mesh from a massive number of minute 3D Gaussian points remains great challenge due to the large volume of Gaussians and difficulty of representation of sharp signals caused by their inherent low-pass characteristics. To address this issue, we propose DyGASR, which utilizes generalized exponential function instead of traditional 3D Gaussian to decrease the number of particles and dynamically optimize the representation of the captured signal. In addition, it is observed that reconstructing mesh with Generalized Exponential Splatting(GES) without modifications frequently leads to failures since the generalized exponential distribution centroids may not precisely align with the scene surface. To overcome this, we adopt Sugar's approach and introduce Generalized Surface Regularization (GSR), which reduces the smallest scaling vector of each point cloud to zero and ensures normal alignment perpendicular to the surface, facilitating subsequent Poisson surface mesh reconstruction. Additionally, we propose a dynamic resolution adjustment strategy that utilizes a cosine schedule to gradually increase image resolution from low to high during the training stage, thus avoiding constant full resolution, which significantly boosts the reconstruction speed. Our approach surpasses existing 3DGS-based mesh reconstruction methods, as evidenced by extensive evaluations on various scene datasets, demonstrating a 25\% increase in speed, and a 30\% reduction in memory usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12471v2">SCIGS: 3D Gaussians Splatting from a Snapshot Compressive Image</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Snapshot Compressive Imaging (SCI) offers a possibility for capturing information in high-speed dynamic scenes, requiring efficient reconstruction method to recover scene information. Despite promising results, current deep learning-based and NeRF-based reconstruction methods face challenges: 1) deep learning-based reconstruction methods struggle to maintain 3D structural consistency within scenes, and 2) NeRF-based reconstruction methods still face limitations in handling dynamic scenes. To address these challenges, we propose SCIGS, a variant of 3DGS, and develop a primitive-level transformation network that utilizes camera pose stamps and Gaussian primitive coordinates as embedding vectors. This approach resolves the necessity of camera pose in vanilla 3DGS and enhances multi-view 3D structural consistency in dynamic scenes by utilizing transformed primitives. Additionally, a high-frequency filter is introduced to eliminate the artifacts generated during the transformation. The proposed SCIGS is the first to reconstruct a 3D explicit scene from a single compressed image, extending its application to dynamic 3D scenes. Experiments on both static and dynamic scenes demonstrate that SCIGS not only enhances SCI decoding but also outperforms current state-of-the-art methods in reconstructing dynamic 3D scenes from a single compressed image. The code will be made available upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16053v1">UnitedVLN: Generalizable Gaussian Splatting for Continuous Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN), where an agent follows instructions to reach a target destination, has recently seen significant advancements. In contrast to navigation in discrete environments with predefined trajectories, VLN in Continuous Environments (VLN-CE) presents greater challenges, as the agent is free to navigate any unobstructed location and is more vulnerable to visual occlusions or blind spots. Recent approaches have attempted to address this by imagining future environments, either through predicted future visual images or semantic features, rather than relying solely on current observations. However, these RGB-based and feature-based methods lack intuitive appearance-level information or high-level semantic complexity crucial for effective navigation. To overcome these limitations, we introduce a novel, generalizable 3DGS-based pre-training paradigm, called UnitedVLN, which enables agents to better explore future environments by unitedly rendering high-fidelity 360 visual images and semantic features. UnitedVLN employs two key schemes: search-then-query sampling and separate-then-united rendering, which facilitate efficient exploitation of neural primitives, helping to integrate both appearance and semantic information for more robust navigation. Extensive experiments demonstrate that UnitedVLN outperforms state-of-the-art methods on existing VLN-CE benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15800v1">PG-SLAM: Photo-realistic and Geometry-aware RGB-D SLAM in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      Simultaneous localization and mapping (SLAM) has achieved impressive performance in static environments. However, SLAM in dynamic environments remains an open question. Many methods directly filter out dynamic objects, resulting in incomplete scene reconstruction and limited accuracy of camera localization. The other works express dynamic objects by point clouds, sparse joints, or coarse meshes, which fails to provide a photo-realistic representation. To overcome the above limitations, we propose a photo-realistic and geometry-aware RGB-D SLAM method by extending Gaussian splatting. Our method is composed of three main modules to 1) map the dynamic foreground including non-rigid humans and rigid items, 2) reconstruct the static background, and 3) localize the camera. To map the foreground, we focus on modeling the deformations and/or motions. We consider the shape priors of humans and exploit geometric and appearance constraints of humans and items. For background mapping, we design an optimization strategy between neighboring local maps by integrating appearance constraint into geometric alignment. As to camera localization, we leverage both static background and dynamic foreground to increase the observations for noise compensation. We explore the geometric and appearance constraints by associating 3D Gaussians with 2D optical flows and pixel patches. Experiments on various real-world datasets demonstrate that our method outperforms state-of-the-art approaches in terms of camera localization and scene representation. Source codes will be publicly available upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15779v1">ZeroGS: Training 3D Gaussian Splatting from Unposed Images</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 16 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Neural radiance fields (NeRF) and 3D Gaussian Splatting (3DGS) are popular techniques to reconstruct and render photo-realistic images. However, the pre-requisite of running Structure-from-Motion (SfM) to get camera poses limits their completeness. While previous methods can reconstruct from a few unposed images, they are not applicable when images are unordered or densely captured. In this work, we propose ZeroGS to train 3DGS from hundreds of unposed and unordered images. Our method leverages a pretrained foundation model as the neural scene representation. Since the accuracy of the predicted pointmaps does not suffice for accurate image registration and high-fidelity image rendering, we propose to mitigate the issue by initializing and finetuning the pretrained model from a seed image. Images are then progressively registered and added to the training buffer, which is further used to train the model. We also propose to refine the camera poses and pointmaps by minimizing a point-to-camera ray consistency loss across multiple views. Experiments on the LLFF dataset, the MipNeRF360 dataset, and the Tanks-and-Temples dataset show that our method recovers more accurate camera poses than state-of-the-art pose-free NeRF/3DGS methods, and even renders higher quality images than 3DGS with COLMAP poses. Our project page is available at https://aibluefisher.github.io/ZeroGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16758v1">Bundle Adjusted Gaussian Avatars Deblurring</a></div>
    <div class="paper-meta">
      📅 2024-11-24
      | 💬 Codes and Data: https://github.com/MyNiuuu/BAGA
    </div>
    <details class="paper-abstract">
      The development of 3D human avatars from multi-view videos represents a significant yet challenging task in the field. Recent advancements, including 3D Gaussian Splattings (3DGS), have markedly progressed this domain. Nonetheless, existing techniques necessitate the use of high-quality sharp images, which are often impractical to obtain in real-world settings due to variations in human motion speed and intensity. In this study, we attempt to explore deriving sharp intrinsic 3D human Gaussian avatars from blurry video footage in an end-to-end manner. Our approach encompasses a 3D-aware, physics-oriented model of blur formation attributable to human movement, coupled with a 3D human motion model to clarify ambiguities found in motion-induced blurry images. This methodology facilitates the concurrent learning of avatar model parameters and the refinement of sub-frame motion parameters from a coarse initialization. We have established benchmarks for this task through a synthetic dataset derived from existing multi-view captures, alongside a real-captured dataset acquired through a 360-degree synchronous hybrid-exposure camera system. Comprehensive evaluations demonstrate that our model surpasses existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03246v6">SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      We present SGS-SLAM, the first semantic visual SLAM system based on Gaussian Splatting. It incorporates appearance, geometry, and semantic features through multi-channel optimization, addressing the oversmoothing limitations of neural implicit SLAM systems in high-quality rendering, scene understanding, and object-level geometry. We introduce a unique semantic feature loss that effectively compensates for the shortcomings of traditional depth and color losses in object optimization. Through a semantic-guided keyframe selection strategy, we prevent erroneous reconstructions caused by cumulative errors. Extensive experiments demonstrate that SGS-SLAM delivers state-of-the-art performance in camera pose estimation, map reconstruction, precise semantic segmentation, and object-level geometric accuracy, while ensuring real-time rendering capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15732v1">DynamicAvatars: Accurate Dynamic Facial Avatars Reconstruction and Precise Editing with Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-11-24
    </div>
    <details class="paper-abstract">
      Generating and editing dynamic 3D head avatars are crucial tasks in virtual reality and film production. However, existing methods often suffer from facial distortions, inaccurate head movements, and limited fine-grained editing capabilities. To address these challenges, we present DynamicAvatars, a dynamic model that generates photorealistic, moving 3D head avatars from video clips and parameters associated with facial positions and expressions. Our approach enables precise editing through a novel prompt-based editing model, which integrates user-provided prompts with guiding parameters derived from large language models (LLMs). To achieve this, we propose a dual-tracking framework based on Gaussian Splatting and introduce a prompt preprocessing module to enhance editing stability. By incorporating a specialized GAN algorithm and connecting it to our control module, which generates precise guiding parameters from LLMs, we successfully address the limitations of existing methods. Additionally, we develop a dynamic editing strategy that selectively utilizes specific training datasets to improve the efficiency and adaptability of the model for dynamic editing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15582v1">EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      Photorealistic reconstruction of street scenes is essential for developing real-world simulators in autonomous driving. While recent methods based on 3D/4D Gaussian Splatting (GS) have demonstrated promising results, they still encounter challenges in complex street scenes due to the unpredictable motion of dynamic objects. Current methods typically decompose street scenes into static and dynamic objects, learning the Gaussians in either a supervised manner (e.g., w/ 3D bounding-box) or a self-supervised manner (e.g., w/o 3D bounding-box). However, these approaches do not effectively model the motions of dynamic objects (e.g., the motion speed of pedestrians is clearly different from that of vehicles), resulting in suboptimal scene decomposition. To address this, we propose Explicit Motion Decomposition (EMD), which models the motions of dynamic objects by introducing learnable motion embeddings to the Gaussians, enhancing the decomposition in street scenes. The proposed EMD is a plug-and-play approach applicable to various baseline methods. We also propose tailored training strategies to apply EMD to both supervised and self-supervised baselines. Through comprehensive experimentation, we illustrate the effectiveness of our approach with various established baselines. The code will be released at: https://qingpowuwu.github.io/emdgaussian.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15482v1">SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      Most existing Dynamic Gaussian Splatting methods for complex dynamic urban scenarios rely on accurate object-level supervision from expensive manual labeling, limiting their scalability in real-world applications. In this paper, we introduce SplatFlow, a Self-Supervised Dynamic Gaussian Splatting within Neural Motion Flow Fields (NMFF) to learn 4D space-time representations without requiring tracked 3D bounding boxes, enabling accurate dynamic scene reconstruction and novel view RGB, depth and flow synthesis. SplatFlow designs a unified framework to seamlessly integrate time-dependent 4D Gaussian representation within NMFF, where NMFF is a set of implicit functions to model temporal motions of both LiDAR points and Gaussians as continuous motion flow fields. Leveraging NMFF, SplatFlow effectively decomposes static background and dynamic objects, representing them with 3D and 4D Gaussian primitives, respectively. NMFF also models the status correspondences of each 4D Gaussian across time, which aggregates temporal features to enhance cross-view consistency of dynamic components. SplatFlow further improves dynamic scene identification by distilling features from 2D foundational models into 4D space-time representation. Comprehensive evaluations conducted on the Waymo Open Dataset and KITTI Dataset validate SplatFlow's state-of-the-art (SOTA) performance for both image reconstruction and novel view synthesis in dynamic urban scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15476v1">Gassidy: Gaussian Splatting SLAM in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2024-11-23
      | 💬 This paper is currently under reviewed for ICRA 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) allows flexible adjustments to scene representation, enabling continuous optimization of scene quality during dense visual simultaneous localization and mapping (SLAM) in static environments. However, 3DGS faces challenges in handling environmental disturbances from dynamic objects with irregular movement, leading to degradation in both camera tracking accuracy and map reconstruction quality. To address this challenge, we develop an RGB-D dense SLAM which is called Gaussian Splatting SLAM in Dynamic Environments (Gassidy). This approach calculates Gaussians to generate rendering loss flows for each environmental component based on a designed photometric-geometric loss function. To distinguish and filter environmental disturbances, we iteratively analyze rendering loss flows to detect features characterized by changes in loss values between dynamic objects and static components. This process ensures a clean environment for accurate scene reconstruction. Compared to state-of-the-art SLAM methods, experimental results on open datasets show that Gassidy improves camera tracking precision by up to 97.9% and enhances map quality by up to 6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15468v1">SplatSDF: Boosting Neural Implicit SDF via Gaussian Splatting Fusion</a></div>
    <div class="paper-meta">
      📅 2024-11-23
    </div>
    <details class="paper-abstract">
      A signed distance function (SDF) is a useful representation for continuous-space geometry and many related operations, including rendering, collision checking, and mesh generation. Hence, reconstructing SDF from image observations accurately and efficiently is a fundamental problem. Recently, neural implicit SDF (SDF-NeRF) techniques, trained using volumetric rendering, have gained a lot of attention. Compared to earlier truncated SDF (TSDF) fusion algorithms that rely on depth maps and voxelize continuous space, SDF-NeRF enables continuous-space SDF reconstruction with better geometric and photometric accuracy. However, the accuracy and convergence speed of scene-level SDF reconstruction require further improvements for many applications. With the advent of 3D Gaussian Splatting (3DGS) as an explicit representation with excellent rendering quality and speed, several works have focused on improving SDF-NeRF by introducing consistency losses on depth and surface normals between 3DGS and SDF-NeRF. However, loss-level connections alone lead to incremental improvements. We propose a novel neural implicit SDF called "SplatSDF" to fuse 3DGSandSDF-NeRF at an architecture level with significant boosts to geometric and photometric accuracy and convergence speed. Our SplatSDF relies on 3DGS as input only during training, and keeps the same complexity and efficiency as the original SDF-NeRF during inference. Our method outperforms state-of-the-art SDF-NeRF models on geometric and photometric evaluation by the time of submission.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17793v2">SafeguardGS: 3D Gaussian Primitive Pruning While Avoiding Catastrophic Scene Destruction</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 19 pages, 20 figures, 7 tables
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in novel view synthesis. However, its suboptimal densification process results in the excessively large number of Gaussian primitives, which impacts frame-per-second and increases memory usage, making it unsuitable for low-end devices. To address this issue, many follow-up studies have proposed various pruning techniques with score functions designed to identify and remove less important primitives. Nonetheless, a comprehensive discussion of their effectiveness and implications across all techniques is missing. In this paper, we are the first to categorize 3DGS pruning techniques into two types: Scene-level pruning and Pixel-level pruning, distinguished by their scope for ranking primitives. Our subsequent experiments reveal that, while scene-level pruning leads to disastrous quality drops under extreme decimation of Gaussian primitives, pixel-level pruning not only sustains relatively high rendering quality with minuscule performance degradation but also provides an inherent boundary of pruning, i.e., a safeguard of Gaussian pruning. Building on this observation, we further propose multiple variations of score functions based on the factors of rendering equations and discover that assessing based on color similarity with blending weight is the most effective method for discriminating insignificant primitives. In our experiments, our SafeguardGS with the optimal score function shows the highest PSNR-per-primitive performance under an extreme pruning setting, retaining only about 10% of the primitives from the original 3DGS scene (i.e., 10x compression ratio). We believe our research provides valuable insights for optimizing 3DGS for future works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13862v2">DepthSplat: Connecting Gaussian Splatting and Depth</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Project page: https://haofeixu.github.io/depthsplat/ Code: https://github.com/cvg/depthsplat
    </div>
    <details class="paper-abstract">
      Gaussian splatting and single/multi-view depth estimation are typically studied in isolation. In this paper, we present DepthSplat to connect Gaussian splatting and depth estimation and study their interactions. More specifically, we first contribute a robust multi-view depth model by leveraging pre-trained monocular depth features, leading to high-quality feed-forward 3D Gaussian splatting reconstructions. We also show that Gaussian splatting can serve as an unsupervised pre-training objective for learning powerful depth models from large-scale unlabeled datasets. We validate the synergy between Gaussian splatting and depth estimation through extensive ablation and cross-task transfer experiments. Our DepthSplat achieves state-of-the-art performance on ScanNet, RealEstate10K and DL3DV datasets in terms of both depth estimation and novel view synthesis, demonstrating the mutual benefits of connecting both tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15355v1">UniGaussian: Driving Scene Reconstruction from Multiple Camera Models via Unified Gaussian Representations</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Urban scene reconstruction is crucial for real-world autonomous driving simulators. Although existing methods have achieved photorealistic reconstruction, they mostly focus on pinhole cameras and neglect fisheye cameras. In fact, how to effectively simulate fisheye cameras in driving scene remains an unsolved problem. In this work, we propose UniGaussian, a novel approach that learns a unified 3D Gaussian representation from multiple camera models for urban scene reconstruction in autonomous driving. Our contributions are two-fold. First, we propose a new differentiable rendering method that distorts 3D Gaussians using a series of affine transformations tailored to fisheye camera models. This addresses the compatibility issue of 3D Gaussian splatting with fisheye cameras, which is hindered by light ray distortion caused by lenses or mirrors. Besides, our method maintains real-time rendering while ensuring differentiability. Second, built on the differentiable rendering method, we design a new framework that learns a unified Gaussian representation from multiple camera models. By applying affine transformations to adapt different camera models and regularizing the shared Gaussians with supervision from different modalities, our framework learns a unified 3D Gaussian representation with input data from multiple sources and achieves holistic driving scene understanding. As a result, our approach models multiple sensors (pinhole and fisheye cameras) and modalities (depth, semantic, normal and LiDAR point clouds). Our experiments show that our method achieves superior rendering quality and fast rendering speed for driving scene simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01593v2">MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Project Page: see https://wcwac.github.io/MaGS-page/
    </div>
    <details class="paper-abstract">
      3D reconstruction and simulation, although interrelated, have distinct objectives: reconstruction requires a flexible 3D representation that can adapt to diverse scenes, while simulation needs a structured representation to model motion principles effectively. This paper introduces the Mesh-adsorbed Gaussian Splatting (MaGS) method to address this challenge. MaGS constrains 3D Gaussians to roam near the mesh, creating a mutually adsorbed mesh-Gaussian 3D representation. Such representation harnesses both the rendering flexibility of 3D Gaussians and the structured property of meshes. To achieve this, we introduce RMD-Net, a network that learns motion priors from video data to refine mesh deformations, alongside RGD-Net, which models the relative displacement between the mesh and Gaussians to enhance rendering fidelity under mesh constraints. To generalize to novel, user-defined deformations beyond input video without reliance on temporal data, we propose MPE-Net, which leverages inherent mesh information to bootstrap RMD-Net and RGD-Net. Due to the universality of meshes, MaGS is compatible with various deformation priors such as ARAP, SMPL, and soft physics simulation. Extensive experiments on the D-NeRF, DG-Mesh, and PeopleSnapshot datasets demonstrate that MaGS achieves state-of-the-art performance in both reconstruction and simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15018v1">Neural 4D Evolution under Large Topological Changes from 2D Images</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 15 pages, 21 figures
    </div>
    <details class="paper-abstract">
      In the literature, it has been shown that the evolution of the known explicit 3D surface to the target one can be learned from 2D images using the instantaneous flow field, where the known and target 3D surfaces may largely differ in topology. We are interested in capturing 4D shapes whose topology changes largely over time. We encounter that the straightforward extension of the existing 3D-based method to the desired 4D case performs poorly. In this work, we address the challenges in extending 3D neural evolution to 4D under large topological changes by proposing two novel modifications. More precisely, we introduce (i) a new architecture to discretize and encode the deformation and learn the SDF and (ii) a technique to impose the temporal consistency. (iii) Also, we propose a rendering scheme for color prediction based on Gaussian splatting. Furthermore, to facilitate learning directly from 2D images, we propose a learning framework that can disentangle the geometry and appearance from RGB images. This method of disentanglement, while also useful for the 4D evolution problem that we are concentrating on, is also novel and valid for static scenes. Our extensive experiments on various data provide awesome results and, most importantly, open a new approach toward reconstructing challenging scenes with significant topological changes and deformations. Our source code and the dataset are publicly available at https://github.com/insait-institute/N4DE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08508v2">BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      We present billboard Splatting (BBSplat) - a novel approach for 3D scene representation based on textured geometric primitives. BBSplat represents the scene as a set of optimizable textured planar primitives with learnable RGB textures and alpha-maps to control their shape. BBSplat primitives can be used in any Gaussian Splatting pipeline as drop-in replacements for Gaussians. Our method's qualitative and quantitative improvements over 3D and 2D Gaussians are most noticeable when fewer primitives are used, when BBSplat achieves over 1200 FPS. Our novel regularization term encourages textures to have a sparser structure, unlocking an efficient compression that leads to a reduction in storage space of the model. Our experiments show the efficiency of BBSplat on standard datasets of real indoor and outdoor scenes such as Tanks&Temples, DTU, and Mip-NeRF-360. We demonstrate improvements on PSNR, SSIM, and LPIPS metrics compared to the state-of-the-art, especially for the case when fewer primitives are used, which, on the other hand, leads to up to 2 times inference speed improvement for the same rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12168v2">Sketch-guided Cage-based 3D Gaussian Splatting Deformation</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 10 pages, 9 figures, project page: https://tianhaoxie.github.io/project/gs_deform/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (GS) is one of the most promising novel 3D representations that has received great interest in computer graphics and computer vision. While various systems have introduced editing capabilities for 3D GS, such as those guided by text prompts, fine-grained control over deformation remains an open challenge. In this work, we present a novel sketch-guided 3D GS deformation system that allows users to intuitively modify the geometry of a 3D GS model by drawing a silhouette sketch from a single viewpoint. Our approach introduces a new deformation method that combines cage-based deformations with a variant of Neural Jacobian Fields, enabling precise, fine-grained control. Additionally, it leverages large-scale 2D diffusion priors and ControlNet to ensure the generated deformations are semantically plausible. Through a series of experiments, we demonstrate the effectiveness of our method and showcase its ability to animate static 3D GS models as one of its key applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07329v4">Cinematic Gaussians: Real-Time HDR Radiance Fields with Depth of Field</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      Radiance field methods represent the state of the art in reconstructing complex scenes from multi-view photos. However, these reconstructions often suffer from one or both of the following limitations: First, they typically represent scenes in low dynamic range (LDR), which restricts their use to evenly lit environments and hinders immersive viewing experiences. Secondly, their reliance on a pinhole camera model, assuming all scene elements are in focus in the input images, presents practical challenges and complicates refocusing during novel-view synthesis. Addressing these limitations, we present a lightweight method based on 3D Gaussian Splatting that utilizes multi-view LDR images of a scene with varying exposure times, apertures, and focus distances as input to reconstruct a high-dynamic-range (HDR) radiance field. By incorporating analytical convolutions of Gaussians based on a thin-lens camera model as well as a tonemapping module, our reconstructions enable the rendering of HDR content with flexible refocusing capabilities. We demonstrate that our combined treatment of HDR and depth of field facilitates real-time cinematic rendering, outperforming the state of the art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14847v1">Dynamics-Aware Gaussian Splatting Streaming Towards Fast On-the-Fly Training for 4D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-22
      | 💬 Project page: https://www.liuzhening.top/DASS
    </div>
    <details class="paper-abstract">
      The recent development of 3D Gaussian Splatting (3DGS) has led to great interest in 4D dynamic spatial reconstruction from multi-view visual inputs. While existing approaches mainly rely on processing full-length multi-view videos for 4D reconstruction, there has been limited exploration of iterative online reconstruction methods that enable on-the-fly training and per-frame streaming. Current 3DGS-based streaming methods treat the Gaussian primitives uniformly and constantly renew the densified Gaussians, thereby overlooking the difference between dynamic and static features and also neglecting the temporal continuity in the scene. To address these limitations, we propose a novel three-stage pipeline for iterative streamable 4D dynamic spatial reconstruction. Our pipeline comprises a selective inheritance stage to preserve temporal continuity, a dynamics-aware shift stage for distinguishing dynamic and static primitives and optimizing their movements, and an error-guided densification stage to accommodate emerging objects. Our method achieves state-of-the-art performance in online 4D reconstruction, demonstrating a 20% improvement in on-the-fly training speed, superior representation quality, and real-time rendering capability. Project page: https://www.liuzhening.top/DASS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00144v2">Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated remarkable effectiveness for novel view synthesis (NVS). However, the 3DGS model tends to overfit when trained with sparse posed views, limiting its generalization ability to novel views. In this paper, we alleviate the overfitting problem, presenting a Self-Ensembling Gaussian Splatting (SE-GS) approach. Our method encompasses a $\mathbf{\Sigma}$-model and a $\mathbf{\Delta}$-model. The $\mathbf{\Sigma}$-model serves as an ensemble of 3DGS models that generates novel-view images during inference. We achieve the self-ensembling by introducing an uncertainty-aware perturbation strategy at the training state. We complement the $\mathbf{\Sigma}$-model with the $\mathbf{\Delta}$-model, which is dynamically perturbed based on the uncertainties of novel-view renderings across different training steps. The perturbation yields diverse temporal samples in the Gaussian parameter space without additional training costs. The geometry of the $\mathbf{\Sigma}$-model is regularized by penalizing discrepancies between the $\mathbf{\Sigma}$-model and these temporal samples. Therefore, our SE-GS conducts an effective and efficient regularization across a large number of 3DGS models, resulting in a robust ensemble, the $\mathbf{\Sigma}$-model. Our experimental results on the LLFF, Mip-NeRF360, DTU, and MVImgNet datasets show that our approach improves NVS quality with few-shot training views, outperforming existing state-of-the-art methods. The code is released at: https://sailor-z.github.io/projects/SEGS.html.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14716v1">VisionPAD: A Vision-Centric Pre-training Paradigm for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-11-22
    </div>
    <details class="paper-abstract">
      This paper introduces VisionPAD, a novel self-supervised pre-training paradigm designed for vision-centric algorithms in autonomous driving. In contrast to previous approaches that employ neural rendering with explicit depth supervision, VisionPAD utilizes more efficient 3D Gaussian Splatting to reconstruct multi-view representations using only images as supervision. Specifically, we introduce a self-supervised method for voxel velocity estimation. By warping voxels to adjacent frames and supervising the rendered outputs, the model effectively learns motion cues in the sequential data. Furthermore, we adopt a multi-frame photometric consistency approach to enhance geometric perception. It projects adjacent frames to the current frame based on rendered depths and relative poses, boosting the 3D geometric representation through pure image supervision. Extensive experiments on autonomous driving datasets demonstrate that VisionPAD significantly improves performance in 3D object detection, occupancy prediction and map segmentation, surpassing state-of-the-art pre-training strategies by a considerable margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14423v1">Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation</a></div>
    <div class="paper-meta">
      📅 2024-11-21
      | 💬 Homepage: https://zhuomanliu.github.io/PhysFlow/
    </div>
    <details class="paper-abstract">
      Realistic simulation of dynamic scenes requires accurately capturing diverse material properties and modeling complex object interactions grounded in physical principles. However, existing methods are constrained to basic material types with limited predictable parameters, making them insufficient to represent the complexity of real-world materials. We introduce a novel approach that leverages multi-modal foundation models and video diffusion to achieve enhanced 4D dynamic scene simulation. Our method utilizes multi-modal models to identify material types and initialize material parameters through image queries, while simultaneously inferring 3D Gaussian splats for detailed scene representation. We further refine these material parameters using video diffusion with a differentiable Material Point Method (MPM) and optical flow guidance rather than render loss or Score Distillation Sampling (SDS) loss. This integrated framework enables accurate prediction and realistic simulation of dynamic interactions in real-world scenarios, advancing both accuracy and flexibility in physics-based simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12089v2">FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-21
    </div>
    <details class="paper-abstract">
      In the real world, objects reveal internal textures when sliced or cut, yet this behavior is not well-studied in 3D generation tasks today. For example, slicing a virtual 3D watermelon should reveal flesh and seeds. Given that no available dataset captures an object's full internal structure and collecting data from all slices is impractical, generative methods become the obvious approach. However, current 3D generation and inpainting methods often focus on visible appearance and overlook internal textures. To bridge this gap, we introduce FruitNinja, the first method to generate internal textures for 3D objects undergoing geometric and topological changes. Our approach produces objects via 3D Gaussian Splatting (3DGS) with both surface and interior textures synthesized, enabling real-time slicing and rendering without additional optimization. FruitNinja leverages a pre-trained diffusion model to progressively inpaint cross-sectional views and applies voxel-grid-based smoothing to achieve cohesive textures throughout the object. Our OpaqueAtom GS strategy overcomes 3DGS limitations by employing densely distributed opaque Gaussians, avoiding biases toward larger particles that destabilize training and sharp color transitions for fine-grained textures. Experimental results show that FruitNinja substantially outperforms existing approaches, showcasing unmatched visual quality in real-time rendered internal views across arbitrary geometry manipulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13753v1">FAST-Splat: Fast, Ambiguity-Free Semantics Transfer in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      We present FAST-Splat for fast, ambiguity-free semantic Gaussian Splatting, which seeks to address the main limitations of existing semantic Gaussian Splatting methods, namely: slow training and rendering speeds; high memory usage; and ambiguous semantic object localization. In deriving FAST-Splat , we formulate open-vocabulary semantic Gaussian Splatting as the problem of extending closed-set semantic distillation to the open-set (open-vocabulary) setting, enabling FAST-Splat to provide precise semantic object localization results, even when prompted with ambiguous user-provided natural-language queries. Further, by exploiting the explicit form of the Gaussian Splatting scene representation to the fullest extent, FAST-Splat retains the remarkable training and rendering speeds of Gaussian Splatting. Specifically, while existing semantic Gaussian Splatting methods distill semantics into a separate neural field or utilize neural models for dimensionality reduction, FAST-Splat directly augments each Gaussian with specific semantic codes, preserving the training, rendering, and memory-usage advantages of Gaussian Splatting over neural field methods. These Gaussian-specific semantic codes, together with a hash-table, enable semantic similarity to be measured with open-vocabulary user prompts and further enable FAST-Splat to respond with unambiguous semantic object labels and 3D masks, unlike prior methods. In experiments, we demonstrate that FAST-Splat is 4x to 6x faster to train with a 13x faster data pre-processing step, achieves between 18x to 75x faster rendering speeds, and requires about 3x smaller GPU memory, compared to the best-competing semantic Gaussian Splatting methods. Further, FAST-Splat achieves relatively similar or better semantic segmentation performance compared to existing methods. After the review period, we will provide links to the project website and the codebase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13549v1">Generating 3D-Consistent Videos from Unposed Internet Photos</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      We address the problem of generating videos from unposed internet photos. A handful of input images serve as keyframes, and our model interpolates between them to simulate a path moving between the cameras. Given random images, a model's ability to capture underlying geometry, recognize scene identity, and relate frames in terms of camera position and orientation reflects a fundamental understanding of 3D structure and scene layout. However, existing video models such as Luma Dream Machine fail at this task. We design a self-supervised method that takes advantage of the consistency of videos and variability of multiview internet photos to train a scalable, 3D-aware video model without any 3D annotations such as camera parameters. We validate that our method outperforms all baselines in terms of geometric and appearance consistency. We also show our model benefits applications that enable camera control, such as 3D Gaussian Splatting. Our results suggest that we can scale up scene-level 3D learning using only 2D data such as videos and multiview internet photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14475v3">MagicDrive3D: Controllable 3D Generation for Any-View Rendering in Street Scenes</a></div>
    <div class="paper-meta">
      📅 2024-11-20
      | 💬 Project Page: https://flymin.github.io/magicdrive3d
    </div>
    <details class="paper-abstract">
      While controllable generative models for images and videos have achieved remarkable success, high-quality models for 3D scenes, particularly in unbounded scenarios like autonomous driving, remain underdeveloped due to high data acquisition costs. In this paper, we introduce MagicDrive3D, a novel pipeline for controllable 3D street scene generation that supports multi-condition control, including BEV maps, 3D objects, and text descriptions. Unlike previous methods that reconstruct before training the generative models, MagicDrive3D first trains a video generation model and then reconstructs from the generated data. This innovative approach enables easily controllable generation and static scene acquisition, resulting in high-quality scene reconstruction. To address the minor errors in generated content, we propose deformable Gaussian splatting with monocular depth initialization and appearance modeling to manage exposure discrepancies across viewpoints. Validated on the nuScenes dataset, MagicDrive3D generates diverse, high-quality 3D driving scenes that support any-view rendering and enhance downstream tasks like BEV segmentation. Our results demonstrate the framework's superior performance, showcasing its potential for autonomous driving simulation and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12981v1">GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Gaze estimation encounters generalization challenges when dealing with out-of-distribution data. To address this problem, recent methods use neural radiance fields (NeRF) to generate augmented data. However, existing methods based on NeRF are computationally expensive and lack facial details. 3D Gaussian Splatting (3DGS) has become the prevailing representation of neural fields. While 3DGS has been extensively examined in head avatars, it faces challenges with accurate gaze control and generalization across different subjects. In this work, we propose GazeGaussian, a high-fidelity gaze redirection method that uses a two-stream 3DGS model to represent the face and eye regions separately. By leveraging the unstructured nature of 3DGS, we develop a novel eye representation for rigid eye rotation based on the target gaze direction. To enhance synthesis generalization across various subjects, we integrate an expression-conditional module to guide the neural renderer. Comprehensive experiments show that GazeGaussian outperforms existing methods in rendering speed, gaze redirection accuracy, and facial synthesis across multiple datasets. We also demonstrate that existing gaze estimation methods can leverage GazeGaussian to improve their generalization performance. The code will be available at: https://ucwxb.github.io/GazeGaussian/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13610v1">Video2BEV: Transforming Drone Videos to BEVs for Video-based Geo-localization</a></div>
    <div class="paper-meta">
      📅 2024-11-20
    </div>
    <details class="paper-abstract">
      Existing approaches to drone visual geo-localization predominantly adopt the image-based setting, where a single drone-view snapshot is matched with images from other platforms. Such task formulation, however, underutilizes the inherent video output of the drone and is sensitive to occlusions and environmental constraints. To address these limitations, we formulate a new video-based drone geo-localization task and propose the Video2BEV paradigm. This paradigm transforms the video into a Bird's Eye View (BEV), simplifying the subsequent matching process. In particular, we employ Gaussian Splatting to reconstruct a 3D scene and obtain the BEV projection. Different from the existing transform methods, \eg, polar transform, our BEVs preserve more fine-grained details without significant distortion. To further improve model scalability toward diverse BEVs and satellite figures, our Video2BEV paradigm also incorporates a diffusion-based module for generating hard negative samples, which facilitates discriminative feature learning. To validate our approach, we introduce UniV, a new video-based geo-localization dataset that extends the image-based University-1652 dataset. UniV features flight paths at $30^\circ$ and $45^\circ$ elevation angles with increased frame rates of up to 10 frames per second (FPS). Extensive experiments on the UniV dataset show that our Video2BEV paradigm achieves competitive recall rates and outperforms conventional video-based methods. Compared to other methods, our proposed approach exhibits robustness at lower elevations with more occlusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04680v3">Next Best Sense: Guiding Vision and Touch with FisherRF for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      We propose a framework for active next best view and touch selection for robotic manipulators using 3D Gaussian Splatting (3DGS). 3DGS is emerging as a useful explicit 3D scene representation for robotics, as it has the ability to represent scenes in a both photorealistic and geometrically accurate manner. However, in real-world, online robotic scenes where the number of views is limited given efficiency requirements, random view selection for 3DGS becomes impractical as views are often overlapping and redundant. We address this issue by proposing an end-to-end online training and active view selection pipeline, which enhances the performance of 3DGS in few-view robotics settings. We first elevate the performance of few-shot 3DGS with a novel semantic depth alignment method using Segment Anything Model 2 (SAM2) that we supplement with Pearson depth and surface normal loss to improve color and depth reconstruction of real-world scenes. We then extend FisherRF, a next-best-view selection method for 3DGS, to select views and touch poses based on depth uncertainty. We perform online view selection on a real robot system during live 3DGS training. We motivate our improvements to few-shot GS scenes, and extend depth-based FisherRF to them, where we demonstrate both qualitative and quantitative improvements on challenging robot scenes. For more information, please see our project page at https://arm.stanford.edu/next-best-sense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09467v2">Enhancing Single Image to 3D Generation using Gaussian Splatting and Hybrid Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      3D object generation from a single image involves estimating the full 3D geometry and texture of unseen views from an unposed RGB image captured in the wild. Accurately reconstructing an object's complete 3D structure and texture has numerous applications in real-world scenarios, including robotic manipulation, grasping, 3D scene understanding, and AR/VR. Recent advancements in 3D object generation have introduced techniques that reconstruct an object's 3D shape and texture by optimizing the efficient representation of Gaussian Splatting, guided by pre-trained 2D or 3D diffusion models. However, a notable disparity exists between the training datasets of these models, leading to distinct differences in their outputs. While 2D models generate highly detailed visuals, they lack cross-view consistency in geometry and texture. In contrast, 3D models ensure consistency across different views but often result in overly smooth textures. We propose bridging the gap between 2D and 3D diffusion models to address this limitation by integrating a two-stage frequency-based distillation loss with Gaussian Splatting. Specifically, we leverage geometric priors in the low-frequency spectrum from a 3D diffusion model to maintain consistent geometry and use a 2D diffusion model to refine the fidelity and texture in the high-frequency spectrum of the generated 3D structure, resulting in more detailed and fine-grained outcomes. Our approach enhances geometric consistency and visual quality, outperforming the current SOTA. Additionally, we demonstrate the easy adaptability of our method for efficient object pose estimation and tracking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12510v1">PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Endoscopic procedures are crucial for colorectal cancer diagnosis, and three-dimensional reconstruction of the environment for real-time novel-view synthesis can significantly enhance diagnosis. We present PR-ENDO, a framework that leverages 3D Gaussian Splatting within a physically based, relightable model tailored for the complex acquisition conditions in endoscopy, such as restricted camera rotations and strong view-dependent illumination. By exploiting the connection between the camera and light source, our approach introduces a relighting model to capture the intricate interactions between light and tissue using physically based rendering and MLP. Existing methods often produce artifacts and inconsistencies under these conditions, which PR-ENDO overcomes by incorporating a specialized diffuse MLP that utilizes light angles and normal vectors, achieving stable reconstructions even with limited training camera rotations. We benchmarked our framework using a publicly available dataset and a newly introduced dataset with wider camera rotations. Our methods demonstrated superior image quality compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12789v1">Automated 3D Physical Simulation of Open-world Scene with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D generation models have opened new possibilities for simulating dynamic 3D object movements and customizing behaviors, yet creating this content remains challenging. Current methods often require manual assignment of precise physical properties for simulations or rely on video generation models to predict them, which is computationally intensive. In this paper, we rethink the usage of multi-modal large language model (MLLM) in physics-based simulation, and present Sim Anything, a physics-based approach that endows static 3D objects with interactive dynamics. We begin with detailed scene reconstruction and object-level 3D open-vocabulary segmentation, progressing to multi-view image in-painting. Inspired by human visual reasoning, we propose MLLM-based Physical Property Perception (MLLM-P3) to predict mean physical properties of objects in a zero-shot manner. Based on the mean values and the object's geometry, the Material Property Distribution Prediction model (MPDP) model then estimates the full distribution, reformulating the problem as probability distribution estimation to reduce computational costs. Finally, we simulate objects in an open-world scene with particles sampled via the Physical-Geometric Adaptive Sampling (PGAS) strategy, efficiently capturing complex deformations and significantly reducing computational costs. Extensive experiments and user studies demonstrate our Sim Anything achieves more realistic motion than state-of-the-art methods within 2 minutes on a single GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09663v3">3D-Consistent Human Avatars with Sparse Inputs via Gaussian Splatting and Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Existing approaches for human avatar generation--both NeRF-based and 3D Gaussian Splatting (3DGS) based--struggle with maintaining 3D consistency and exhibit degraded detail reconstruction, particularly when training with sparse inputs. To address this challenge, we propose CHASE, a novel framework that achieves dense-input-level performance using only sparse inputs through two key innovations: cross-pose intrinsic 3D consistency supervision and 3D geometry contrastive learning. Building upon prior skeleton-driven approaches that combine rigid deformation with non-rigid cloth dynamics, we first establish baseline avatars with fundamental 3D consistency. To enhance 3D consistency under sparse inputs, we introduce a Dynamic Avatar Adjustment (DAA) module, which refines deformed Gaussians by leveraging similar poses from the training set. By minimizing the rendering discrepancy between adjusted Gaussians and reference poses, DAA provides additional supervision for avatar reconstruction. We further maintain global 3D consistency through a novel geometry-aware contrastive learning strategy. While designed for sparse inputs, CHASE surpasses state-of-the-art methods across both full and sparse settings on ZJU-MoCap and H36M datasets, demonstrating that our enhanced 3D consistency leads to superior rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09665v2">Topology-aware Human Avatars with Semantically-guided Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Reconstructing photo-realistic and topology-aware animatable human avatars from monocular videos remains challenging in computer vision and graphics. Recently, methods using 3D Gaussians to represent the human body have emerged, offering faster optimization and real-time rendering. However, due to ignoring the crucial role of human body semantic information which represents the explicit topological and intrinsic structure within human body, they fail to achieve fine-detail reconstruction of human avatars. To address this issue, we propose SG-GS, which uses semantics-embedded 3D Gaussians, skeleton-driven rigid deformation, and non-rigid cloth dynamics deformation to create photo-realistic human avatars. We then design a Semantic Human-Body Annotator (SHA) which utilizes SMPL's semantic prior for efficient body part semantic labeling. The generated labels are used to guide the optimization of semantic attributes of Gaussian. To capture the explicit topological structure of the human body, we employ a 3D network that integrates both topological and geometric associations for human avatar deformation. We further implement three key strategies to enhance the semantic accuracy of 3D Gaussians and rendering quality: semantic projection with 2D regularization, semantic-guided density regularization and semantic-aware regularization with neighborhood consistency. Extensive experiments demonstrate that SG-GS achieves state-of-the-art geometry and appearance reconstruction performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15193v1">Gradient-Weighted Feature Back-Projection: A Fast Alternative to Feature Distillation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      We introduce a training-free method for feature field rendering in Gaussian splatting. Our approach back-projects 2D features into pre-trained 3D Gaussians, using a weighted sum based on each Gaussian's influence in the final rendering. While most training-based feature field rendering methods excel at 2D segmentation but perform poorly at 3D segmentation without post-processing, our method achieves high-quality results in both 2D and 3D segmentation. Experimental results demonstrate that our approach is fast, scalable, and offers performance comparable to training-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12788v1">Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      In this study, we explore the essential challenge of fast scene optimization for Gaussian Splatting. Through a thorough analysis of the geometry modeling process, we reveal that dense point clouds can be effectively reconstructed early in optimization through Gaussian representations. This insight leads to our approach of aggressive Gaussian densification, which provides a more efficient alternative to conventional progressive densification methods. By significantly increasing the number of critical Gaussians, we enhance the model capacity to capture dense scene geometry at the early stage of optimization. This strategy is seamlessly integrated into the Mini-Splatting densification and simplification framework, enabling rapid convergence without compromising quality. Additionally, we introduce visibility culling within Gaussian Splatting, leveraging per-view Gaussian importance as precomputed visibility to accelerate the optimization process. Our Mini-Splatting2 achieves a balanced trade-off among optimization time, the number of Gaussians, and rendering quality, establishing a strong baseline for future Gaussian-Splatting-based works. Our work sets the stage for more efficient, high-quality 3D scene modeling in real-world applications, and the code will be made available no matter acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12185v1">LiV-GS: LiDAR-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      We present LiV-GS, a LiDAR-visual SLAM system in outdoor environments that leverages 3D Gaussian as a differentiable spatial representation. Notably, LiV-GS is the first method that directly aligns discrete and sparse LiDAR data with continuous differentiable Gaussian maps in large-scale outdoor scenes, overcoming the limitation of fixed resolution in traditional LiDAR mapping. The system aligns point clouds with Gaussian maps using shared covariance attributes for front-end tracking and integrates the normal orientation into the loss function to refines the Gaussian map. To reliably and stably update Gaussians outside the LiDAR field of view, we introduce a novel conditional Gaussian constraint that aligns these Gaussians closely with the nearest reliable ones. The targeted adjustment enables LiV-GS to achieve fast and accurate mapping with novel view synthesis at a rate of 7.98 FPS. Extensive comparative experiments demonstrate LiV-GS's superior performance in SLAM, image rendering and mapping. The successful cross-modal radar-LiDAR localization highlights the potential of LiV-GS for applications in cross-modal semantic positioning and object segmentation with Gaussian maps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19957v4">PLA4D: Pixel-Level Alignments for Text-to-4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Previous text-to-4D methods have leveraged multiple Score Distillation Sampling (SDS) techniques, combining motion priors from video-based diffusion models (DMs) with geometric priors from multiview DMs to implicitly guide 4D renderings. However, differences in these priors result in conflicting gradient directions during optimization, causing trade-offs between motion fidelity and geometry accuracy, and requiring substantial optimization time to reconcile the models. In this paper, we introduce \textbf{P}ixel-\textbf{L}evel \textbf{A}lignment for text-driven \textbf{4D} Gaussian splatting (PLA4D) to resolve this motion-geometry conflict. PLA4D provides an anchor reference, i.e., text-generated video, to align the rendering process conditioned by different DMs in pixel space. For static alignment, our approach introduces a focal alignment method and Gaussian-Mesh contrastive learning to iteratively adjust focal lengths and provide explicit geometric priors at each timestep. At the dynamic level, a motion alignment technique and T-MV refinement method are employed to enforce both pose alignment and motion continuity across unknown viewpoints, ensuring intrinsic geometric consistency across views. With such pixel-level multi-DM alignment, our PLA4D framework is able to generate 4D objects with superior geometric, motion, and semantic consistency. Fully implemented with open-source tools, PLA4D offers an efficient and accessible solution for high-quality 4D digital content creation with significantly reduced generation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.10722v1">DGS-SLAM: Gaussian Splatting SLAM in Dynamic Environment</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 Preprint, Under review
    </div>
    <details class="paper-abstract">
      We introduce Dynamic Gaussian Splatting SLAM (DGS-SLAM), the first dynamic SLAM framework built on the foundation of Gaussian Splatting. While recent advancements in dense SLAM have leveraged Gaussian Splatting to enhance scene representation, most approaches assume a static environment, making them vulnerable to photometric and geometric inconsistencies caused by dynamic objects. To address these challenges, we integrate Gaussian Splatting SLAM with a robust filtering process to handle dynamic objects throughout the entire pipeline, including Gaussian insertion and keyframe selection. Within this framework, to further improve the accuracy of dynamic object removal, we introduce a robust mask generation method that enforces photometric consistency across keyframes, reducing noise from inaccurate segmentation and artifacts such as shadows. Additionally, we propose the loop-aware window selection mechanism, which utilizes unique keyframe IDs of 3D Gaussians to detect loops between the current and past frames, facilitating joint optimization of the current camera poses and the Gaussian map. DGS-SLAM achieves state-of-the-art performance in both camera tracking and novel view synthesis on various dynamic SLAM benchmarks, proving its effectiveness in handling real-world dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.11024v1">VeGaS: Video Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) employ neural networks to approximate discrete data as continuous functions. In the context of video data, such models can be utilized to transform the coordinates of pixel locations along with frame occurrence times (or indices) into RGB color values. Although INRs facilitate effective compression, they are unsuitable for editing purposes. One potential solution is to use a 3D Gaussian Splatting (3DGS) based model, such as the Video Gaussian Representation (VGR), which is capable of encoding video as a multitude of 3D Gaussians and is applicable for numerous video processing operations, including editing. Nevertheless, in this case, the capacity for modification is constrained to a limited set of basic transformations. To address this issue, we introduce the Video Gaussian Splatting (VeGaS) model, which enables realistic modifications of video data. To construct VeGaS, we propose a novel family of Folded-Gaussian distributions designed to capture nonlinear dynamics in a video stream and model consecutive frames by 2D Gaussians obtained as respective conditional distributions. Our experiments demonstrate that VeGaS outperforms state-of-the-art solutions in frame reconstruction tasks and allows realistic modifications of video data. The code is available at: https://github.com/gmum/VeGaS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11839v1">RoboGSim: A Real2Sim2Real Robotic Gaussian Splatting Simulator</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Efficient acquisition of real-world embodied data has been increasingly critical. However, large-scale demonstrations captured by remote operation tend to take extremely high costs and fail to scale up the data size in an efficient manner. Sampling the episodes under a simulated environment is a promising way for large-scale collection while existing simulators fail to high-fidelity modeling on texture and physics. To address these limitations, we introduce the RoboGSim, a real2sim2real robotic simulator, powered by 3D Gaussian Splatting and the physics engine. RoboGSim mainly includes four parts: Gaussian Reconstructor, Digital Twins Builder, Scene Composer, and Interactive Engine. It can synthesize the simulated data with novel views, objects, trajectories, and scenes. RoboGSim also provides an online, reproducible, and safe evaluation for different manipulation policies. The real2sim and sim2real transfer experiments show a high consistency in the texture and physics. Moreover, the effectiveness of synthetic data is validated under the real-world manipulated tasks. We hope RoboGSim serves as a closed-loop simulator for fair comparison on policy learning. More information can be found on our project page https://robogsim.github.io/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11941v1">TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Dynamic scene reconstruction is a long-term challenge in 3D vision. Recent methods extend 3D Gaussian Splatting to dynamic scenes via additional deformation fields and apply explicit constraints like motion flow to guide the deformation. However, they learn motion changes from individual timestamps independently, making it challenging to reconstruct complex scenes, particularly when dealing with violent movement, extreme-shaped geometries, or reflective surfaces. To address the above issue, we design a plug-and-play module called TimeFormer to enable existing deformable 3D Gaussians reconstruction methods with the ability to implicitly model motion patterns from a learning perspective. Specifically, TimeFormer includes a Cross-Temporal Transformer Encoder, which adaptively learns the temporal relationships of deformable 3D Gaussians. Furthermore, we propose a two-stream optimization strategy that transfers the motion knowledge learned from TimeFormer to the base stream during the training phase. This allows us to remove TimeFormer during inference, thereby preserving the original rendering speed. Extensive experiments in the multi-view and monocular dynamic scenes validate qualitative and quantitative improvement brought by TimeFormer. Project Page: https://patrickddj.github.io/TimeFormer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11273v2">BrightDreamer: Generic 3D Gaussian Generative Framework for Fast Text-to-3D Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Text-to-3D synthesis has recently seen intriguing advances by combining the text-to-image priors with 3D representation methods, e.g., 3D Gaussian Splatting (3D GS), via Score Distillation Sampling (SDS). However, a hurdle of existing methods is the low efficiency, per-prompt optimization for a single 3D object. Therefore, it is imperative for a paradigm shift from per-prompt optimization to feed-forward generation for any unseen text prompts, which yet remains challenging. An obstacle is how to directly generate a set of millions of 3D Gaussians to represent a 3D object. This paper presents BrightDreamer, an end-to-end feed-forward approach that can achieve generalizable and fast (77 ms) text-to-3D generation. Our key idea is to formulate the generation process as estimating the 3D deformation from an anchor shape with predefined positions. For this, we first propose a Text-guided Shape Deformation (TSD) network to predict the deformed shape and its new positions, used as the centers (one attribute) of 3D Gaussians. To estimate the other four attributes (i.e., scaling, rotation, opacity, and SH), we then design a novel Text-guided Triplane Generator (TTG) to generate a triplane representation for a 3D object. The center of each Gaussian enables us to transform the spatial feature into the four attributes. The generated 3D Gaussians can be finally rendered at 705 frames per second. Extensive experiments demonstrate the superiority of our method over existing methods. Also, BrightDreamer possesses a strong semantic understanding capability even for complex text prompts. The code is available in the project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11363v1">GPS-Gaussian+: Generalizable Pixel-wise 3D Gaussian Splatting for Real-Time Human-Scene Rendering from Sparse Views</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 Journal extension of CVPR 2024,Project page:https://yaourtb.github.io/GPS-Gaussian+
    </div>
    <details class="paper-abstract">
      Differentiable rendering techniques have recently shown promising results for free-viewpoint video synthesis of characters. However, such methods, either Gaussian Splatting or neural implicit rendering, typically necessitate per-subject optimization which does not meet the requirement of real-time rendering in an interactive application. We propose a generalizable Gaussian Splatting approach for high-resolution image rendering under a sparse-view camera setting. To this end, we introduce Gaussian parameter maps defined on the source views and directly regress Gaussian properties for instant novel view synthesis without any fine-tuning or optimization. We train our Gaussian parameter regression module on human-only data or human-scene data, jointly with a depth estimation module to lift 2D parameter maps to 3D space. The proposed framework is fully differentiable with both depth and rendering supervision or with only rendering supervision. We further introduce a regularization term and an epipolar attention mechanism to preserve geometry consistency between two source views, especially when neglecting depth supervision. Experiments on several datasets demonstrate that our method outperforms state-of-the-art methods while achieving an exceeding rendering speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11921v1">DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      We present DeSiRe-GS, a self-supervised gaussian splatting representation, enabling effective static-dynamic decomposition and high-fidelity surface reconstruction in complex driving scenarios. Our approach employs a two-stage optimization pipeline of dynamic street Gaussians. In the first stage, we extract 2D motion masks based on the observation that 3D Gaussian Splatting inherently can reconstruct only the static regions in dynamic environments. These extracted 2D motion priors are then mapped into the Gaussian space in a differentiable manner, leveraging an efficient formulation of dynamic Gaussians in the second stage. Combined with the introduced geometric regularizations, our method are able to address the over-fitting issues caused by data sparsity in autonomous driving, reconstructing physically plausible Gaussians that align with object surfaces rather than floating in air. Furthermore, we introduce temporal cross-view consistency to ensure coherence across time and viewpoints, resulting in high-quality surface reconstruction. Comprehensive experiments demonstrate the efficiency and effectiveness of DeSiRe-GS, surpassing prior self-supervised arts and achieving accuracy comparable to methods relying on external 3D bounding box annotations. Code is available at \url{https://github.com/chengweialan/DeSiRe-GS}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11024v1">VeGaS: Video Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-11-17
    </div>
    <details class="paper-abstract">
      Implicit Neural Representations (INRs) employ neural networks to approximate discrete data as continuous functions. In the context of video data, such models can be utilized to transform the coordinates of pixel locations along with frame occurrence times (or indices) into RGB color values. Although INRs facilitate effective compression, they are unsuitable for editing purposes. One potential solution is to use a 3D Gaussian Splatting (3DGS) based model, such as the Video Gaussian Representation (VGR), which is capable of encoding video as a multitude of 3D Gaussians and is applicable for numerous video processing operations, including editing. Nevertheless, in this case, the capacity for modification is constrained to a limited set of basic transformations. To address this issue, we introduce the Video Gaussian Splatting (VeGaS) model, which enables realistic modifications of video data. To construct VeGaS, we propose a novel family of Folded-Gaussian distributions designed to capture nonlinear dynamics in a video stream and model consecutive frames by 2D Gaussians obtained as respective conditional distributions. Our experiments demonstrate that VeGaS outperforms state-of-the-art solutions in frame reconstruction tasks and allows realistic modifications of video data. The code is available at: https://github.com/gmum/VeGaS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10947v1">Direct and Explicit 3D Generation from a Single Image</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 3DV 2025, Project page: https://hao-yu-wu.github.io/gen3d/
    </div>
    <details class="paper-abstract">
      Current image-to-3D approaches suffer from high computational costs and lack scalability for high-resolution outputs. In contrast, we introduce a novel framework to directly generate explicit surface geometry and texture using multi-view 2D depth and RGB images along with 3D Gaussian features using a repurposed Stable Diffusion model. We introduce a depth branch into U-Net for efficient and high quality multi-view, cross-domain generation and incorporate epipolar attention into the latent-to-pixel decoder for pixel-level multi-view consistency. By back-projecting the generated depth pixels into 3D space, we create a structured 3D representation that can be either rendered via Gaussian splatting or extracted to high-quality meshes, thereby leveraging additional novel view synthesis loss to further improve our performance. Extensive experiments demonstrate that our method surpasses existing baselines in geometry and texture quality while achieving significantly faster generation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10722v1">DGS-SLAM: Gaussian Splatting SLAM in Dynamic Environment</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 Preprint, Under review
    </div>
    <details class="paper-abstract">
      We introduce Dynamic Gaussian Splatting SLAM (DGS-SLAM), the first dynamic SLAM framework built on the foundation of Gaussian Splatting. While recent advancements in dense SLAM have leveraged Gaussian Splatting to enhance scene representation, most approaches assume a static environment, making them vulnerable to photometric and geometric inconsistencies caused by dynamic objects. To address these challenges, we integrate Gaussian Splatting SLAM with a robust filtering process to handle dynamic objects throughout the entire pipeline, including Gaussian insertion and keyframe selection. Within this framework, to further improve the accuracy of dynamic object removal, we introduce a robust mask generation method that enforces photometric consistency across keyframes, reducing noise from inaccurate segmentation and artifacts such as shadows. Additionally, we propose the loop-aware window selection mechanism, which utilizes unique keyframe IDs of 3D Gaussians to detect loops between the current and past frames, facilitating joint optimization of the current camera poses and the Gaussian map. DGS-SLAM achieves state-of-the-art performance in both camera tracking and novel view synthesis on various dynamic SLAM benchmarks, proving its effectiveness in handling real-world dynamic scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02968v2">GSGAN: Adversarial Learning for Hierarchical Generation of 3D Gaussian Splats</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 NeurIPS 2024 / Project page: https://hse1032.github.io/gsgan
    </div>
    <details class="paper-abstract">
      Most advances in 3D Generative Adversarial Networks (3D GANs) largely depend on ray casting-based volume rendering, which incurs demanding rendering costs. One promising alternative is rasterization-based 3D Gaussian Splatting (3D-GS), providing a much faster rendering speed and explicit 3D representation. In this paper, we exploit Gaussian as a 3D representation for 3D GANs by leveraging its efficient and explicit characteristics. However, in an adversarial framework, we observe that a na\"ive generator architecture suffers from training instability and lacks the capability to adjust the scale of Gaussians. This leads to model divergence and visual artifacts due to the absence of proper guidance for initialized positions of Gaussians and densification to manage their scales adaptively. To address these issues, we introduce a generator architecture with a hierarchical multi-scale Gaussian representation that effectively regularizes the position and scale of generated Gaussians. Specifically, we design a hierarchy of Gaussians where finer-level Gaussians are parameterized by their coarser-level counterparts; the position of finer-level Gaussians would be located near their coarser-level counterparts, and the scale would monotonically decrease as the level becomes finer, modeling both coarse and fine details of the 3D scene. Experimental results demonstrate that ours achieves a significantly faster rendering speed (x100) compared to state-of-the-art 3D consistent GANs with comparable 3D generation capability. Project page: https://hse1032.github.io/gsgan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12592v1">SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Recent efforts in Gaussian-Splat-based Novel View Synthesis can achieve photorealistic rendering; however, such capability is limited in sparse-view scenarios due to sparse initialization and over-fitting floaters. Recent progress in depth estimation and alignment can provide dense point cloud with few views; however, the resulting pose accuracy is suboptimal. In this work, we present SPARS3R, which combines the advantages of accurate pose estimation from Structure-from-Motion and dense point cloud from depth estimation. To this end, SPARS3R first performs a Global Fusion Alignment process that maps a prior dense point cloud to a sparse point cloud from Structure-from-Motion based on triangulated correspondences. RANSAC is applied during this process to distinguish inliers and outliers. SPARS3R then performs a second, Semantic Outlier Alignment step, which extracts semantically coherent regions around the outliers and performs local alignment in these regions. Along with several improvements in the evaluation process, we demonstrate that SPARS3R can achieve photorealistic rendering with sparse images and significantly outperforms existing approaches.
    </details>
</div>
