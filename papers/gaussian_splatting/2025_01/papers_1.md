# gaussian splatting - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17978v2">VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-31
    </div>
    <details class="paper-abstract">
      Reconstructing a 3D scene from images is challenging due to the different ways light interacts with surfaces depending on the viewer's position and the surface's material. In classical computer graphics, materials can be classified as diffuse or specular, interacting with light differently. The standard 3D Gaussian Splatting model struggles to represent view-dependent content, since it cannot differentiate an object within the scene from the light interacting with its specular surfaces, which produce highlights or reflections. In this paper, we propose to extend the 3D Gaussian Splatting model by introducing an additional symmetric matrix to enhance the opacity representation of each 3D Gaussian. This improvement allows certain Gaussians to be suppressed based on the viewer's perspective, resulting in a more accurate representation of view-dependent reflections and specular highlights without compromising the scene's integrity. By allowing the opacity to be view dependent, our enhanced model achieves state-of-the-art performance on Mip-Nerf, Tanks&Temples, Deep Blending, and Nerf-Synthetic datasets without a significant loss in rendering speed, achieving >60FPS, and only incurring a minimal increase in memory used.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19319v1">Advancing Dense Endoscopic Reconstruction with Gaussian Splatting-driven Surface Normal-aware Tracking and Mapping</a></div>
    <div class="paper-meta">
      📅 2025-01-31
      | 💬 Accepted by ICRA 2025
    </div>
    <details class="paper-abstract">
      Simultaneous Localization and Mapping (SLAM) is essential for precise surgical interventions and robotic tasks in minimally invasive procedures. While recent advancements in 3D Gaussian Splatting (3DGS) have improved SLAM with high-quality novel view synthesis and fast rendering, these systems struggle with accurate depth and surface reconstruction due to multi-view inconsistencies. Simply incorporating SLAM and 3DGS leads to mismatches between the reconstructed frames. In this work, we present Endo-2DTAM, a real-time endoscopic SLAM system with 2D Gaussian Splatting (2DGS) to address these challenges. Endo-2DTAM incorporates a surface normal-aware pipeline, which consists of tracking, mapping, and bundle adjustment modules for geometrically accurate reconstruction. Our robust tracking module combines point-to-point and point-to-plane distance metrics, while the mapping module utilizes normal consistency and depth distortion to enhance surface reconstruction quality. We also introduce a pose-consistent strategy for efficient and geometrically coherent keyframe sampling. Extensive experiments on public endoscopic datasets demonstrate that Endo-2DTAM achieves an RMSE of $1.87\pm 0.63$ mm for depth reconstruction of surgical scenes while maintaining computationally efficient tracking, high-quality visual appearance, and real-time rendering. Our code will be released at github.com/lastbasket/Endo-2DTAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19196v1">RaySplats: Ray Tracing based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a process that enables the direct creation of 3D objects from 2D images. This representation offers numerous advantages, including rapid training and rendering. However, a significant limitation of 3DGS is the challenge of incorporating light and shadow reflections, primarily due to the utilization of rasterization rather than ray tracing for rendering. This paper introduces RaySplats, a model that employs ray-tracing based Gaussian Splatting. Rather than utilizing the projection of Gaussians, our method employs a ray-tracing mechanism, operating directly on Gaussian primitives represented by confidence ellipses with RGB colors. In practice, we compute the intersection between ellipses and rays to construct ray-tracing algorithms, facilitating the incorporation of meshes with Gaussian Splatting models and the addition of lights, shadows, and other related effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19088v1">JGHand: Joint-Driven Animatable Hand Avater via 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-31
    </div>
    <details class="paper-abstract">
      Since hands are the primary interface in daily interactions, modeling high-quality digital human hands and rendering realistic images is a critical research problem. Furthermore, considering the requirements of interactive and rendering applications, it is essential to achieve real-time rendering and driveability of the digital model without compromising rendering quality. Thus, we propose Jointly 3D Gaussian Hand (JGHand), a novel joint-driven 3D Gaussian Splatting (3DGS)-based hand representation that renders high-fidelity hand images in real-time for various poses and characters. Distinct from existing articulated neural rendering techniques, we introduce a differentiable process for spatial transformations based on 3D key points. This process supports deformations from the canonical template to a mesh with arbitrary bone lengths and poses. Additionally, we propose a real-time shadow simulation method based on per-pixel depth to simulate self-occlusion shadows caused by finger movements. Finally, we embed the hand prior and propose an animatable 3DGS representation of the hand driven solely by 3D key points. We validate the effectiveness of each component of our approach through comprehensive ablation studies. Experimental results on public datasets demonstrate that JGHand achieves real-time rendering speeds with enhanced quality, surpassing state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00173v1">Lifting by Gaussians: A Simple, Fast and Flexible Method for 3D Instance Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-01-31
      | 💬 Accepted to WACV 2025
    </div>
    <details class="paper-abstract">
      We introduce Lifting By Gaussians (LBG), a novel approach for open-world instance segmentation of 3D Gaussian Splatted Radiance Fields (3DGS). Recently, 3DGS Fields have emerged as a highly efficient and explicit alternative to Neural Field-based methods for high-quality Novel View Synthesis. Our 3D instance segmentation method directly lifts 2D segmentation masks from SAM (alternately FastSAM, etc.), together with features from CLIP and DINOv2, directly fusing them onto 3DGS (or similar Gaussian radiance fields such as 2DGS). Unlike previous approaches, LBG requires no per-scene training, allowing it to operate seamlessly on any existing 3DGS reconstruction. Our approach is not only an order of magnitude faster and simpler than existing approaches; it is also highly modular, enabling 3D semantic segmentation of existing 3DGS fields without requiring a specific parametrization of the 3D Gaussians. Furthermore, our technique achieves superior semantic segmentation for 2D semantic novel view synthesis and 3D asset extraction results while maintaining flexibility and efficiency. We further introduce a novel approach to evaluate individually segmented 3D assets from 3D radiance field segmentation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.19196v1">RaySplats: Ray Tracing based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-31
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is a process that enables the direct creation of 3D objects from 2D images. This representation offers numerous advantages, including rapid training and rendering. However, a significant limitation of 3DGS is the challenge of incorporating light and shadow reflections, primarily due to the utilization of rasterization rather than ray tracing for rendering. This paper introduces RaySplats, a model that employs ray-tracing based Gaussian Splatting. Rather than utilizing the projection of Gaussians, our method employs a ray-tracing mechanism, operating directly on Gaussian primitives represented by confidence ellipses with RGB colors. In practice, we compute the intersection between ellipses and rays to construct ray-tracing algorithms, facilitating the incorporation of meshes with Gaussian Splatting models and the addition of lights, shadows, and other related effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18672v1">Drag Your Gaussian: Effective Drag-Based Editing with Score Distillation for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-30
      | 💬 Visit our project page at https://quyans.github.io/Drag-Your-Gaussian
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D scene editing have been propelled by the rapid development of generative models. Existing methods typically utilize generative models to perform text-guided editing on 3D representations, such as 3D Gaussian Splatting (3DGS). However, these methods are often limited to texture modifications and fail when addressing geometric changes, such as editing a character's head to turn around. Moreover, such methods lack accurate control over the spatial position of editing results, as language struggles to precisely describe the extent of edits. To overcome these limitations, we introduce DYG, an effective 3D drag-based editing method for 3D Gaussian Splatting. It enables users to conveniently specify the desired editing region and the desired dragging direction through the input of 3D masks and pairs of control points, thereby enabling precise control over the extent of editing. DYG integrates the strengths of the implicit triplane representation to establish the geometric scaffold of the editing results, effectively overcoming suboptimal editing outcomes caused by the sparsity of 3DGS in the desired editing regions. Additionally, we incorporate a drag-based Latent Diffusion Model into our method through the proposed Drag-SDS loss function, enabling flexible, multi-view consistent, and fine-grained editing. Extensive experiments demonstrate that DYG conducts effective drag-based editing guided by control point prompts, surpassing other baselines in terms of editing effect and quality, both qualitatively and quantitatively. Visit our project page at https://quyans.github.io/Drag-Your-Gaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09295v2">GEVO: Memory-Efficient Monocular Visual Odometry Using Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Constructing a high-fidelity representation of the 3D scene using a monocular camera can enable a wide range of applications on mobile devices, such as micro-robots, smartphones, and AR/VR headsets. On these devices, memory is often limited in capacity and its access often dominates the consumption of compute energy. Although Gaussian Splatting (GS) allows for high-fidelity reconstruction of 3D scenes, current GS-based SLAM is not memory efficient as a large number of past images is stored to retrain Gaussians for reducing catastrophic forgetting. These images often require two-orders-of-magnitude higher memory than the map itself and thus dominate the total memory usage. In this work, we present GEVO, a GS-based monocular SLAM framework that achieves comparable fidelity as prior methods by rendering (instead of storing) them from the existing map. Novel Gaussian initialization and optimization techniques are proposed to remove artifacts from the map and delay the degradation of the rendered images over time. Across a variety of environments, GEVO achieves comparable map fidelity while reducing the memory overhead to around 58 MBs, which is up to 94x lower than prior works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17978v1">VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      Reconstructing a 3D scene from images is challenging due to the different ways light interacts with surfaces depending on the viewer's position and the surface's material. In classical computer graphics, materials can be classified as diffuse or specular, interacting with light differently. The standard 3D Gaussian Splatting model struggles to represent view-dependent content, since it cannot differentiate an object within the scene from the light interacting with its specular surfaces, which produce highlights or reflections. In this paper, we propose to extend the 3D Gaussian Splatting model by introducing an additional symmetric matrix to enhance the opacity representation of each 3D Gaussian. This improvement allows certain Gaussians to be suppressed based on the viewer's perspective, resulting in a more accurate representation of view-dependent reflections and specular highlights without compromising the scene's integrity. By allowing the opacity to be view dependent, our enhanced model achieves state-of-the-art performance on Mip-Nerf, Tanks\&Temples, Deep Blending, and Nerf-Synthetic datasets without a significant loss in rendering speed, achieving >60FPS, and only incurring a minimal increase in memory used.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17792v1">CrowdSplat: Exploring Gaussian Splatting For Crowd Rendering</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 4 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We present CrowdSplat, a novel approach that leverages 3D Gaussian Splatting for real-time, high-quality crowd rendering. Our method utilizes 3D Gaussian functions to represent animated human characters in diverse poses and outfits, which are extracted from monocular videos. We integrate Level of Detail (LoD) rendering to optimize computational efficiency and quality. The CrowdSplat framework consists of two stages: (1) avatar reconstruction and (2) crowd synthesis. The framework is also optimized for GPU memory usage to enhance scalability. Quantitative and qualitative evaluations show that CrowdSplat achieves good levels of rendering quality, memory efficiency, and computational performance. Through these experiments, we demonstrate that CrowdSplat is a viable solution for dynamic, realistic crowd simulation in real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17655v1">FeatureGS: Eigenvalue-Feature Optimization in 3D Gaussian Splatting for Geometrically Accurate and Artifact-Reduced Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-01-29
      | 💬 16 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful approach for 3D scene reconstruction using 3D Gaussians. However, neither the centers nor surfaces of the Gaussians are accurately aligned to the object surface, complicating their direct use in point cloud and mesh reconstruction. Additionally, 3DGS typically produces floater artifacts, increasing the number of Gaussians and storage requirements. To address these issues, we present FeatureGS, which incorporates an additional geometric loss term based on an eigenvalue-derived 3D shape feature into the optimization process of 3DGS. The goal is to improve geometric accuracy and enhance properties of planar surfaces with reduced structural entropy in local 3D neighborhoods.We present four alternative formulations for the geometric loss term based on 'planarity' of Gaussians, as well as 'planarity', 'omnivariance', and 'eigenentropy' of Gaussian neighborhoods. We provide quantitative and qualitative evaluations on 15 scenes of the DTU benchmark dataset focusing on following key aspects: Geometric accuracy and artifact-reduction, measured by the Chamfer distance, and memory efficiency, evaluated by the total number of Gaussians. Additionally, rendering quality is monitored by Peak Signal-to-Noise Ratio. FeatureGS achieves a 30 % improvement in geometric accuracy, reduces the number of Gaussians by 90 %, and suppresses floater artifacts, while maintaining comparable photometric rendering quality. The geometric loss with 'planarity' from Gaussians provides the highest geometric accuracy, while 'omnivariance' in Gaussian neighborhoods reduces floater artifacts and number of Gaussians the most. This makes FeatureGS a strong method for geometrically accurate, artifact-reduced and memory-efficient 3D scene reconstruction, enabling the direct use of Gaussian centers for geometric representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18643v1">3D Reconstruction of Shoes for Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-01-29
    </div>
    <details class="paper-abstract">
      This paper introduces a mobile-based solution that enhances online shoe shopping through 3D modeling and Augmented Reality (AR), leveraging the efficiency of 3D Gaussian Splatting. Addressing the limitations of static 2D images, the framework generates realistic 3D shoe models from 2D images, achieving an average Peak Signal-to-Noise Ratio (PSNR) of 0.32, and enables immersive AR interactions via smartphones. A custom shoe segmentation dataset of 3120 images was created, with the best-performing segmentation model achieving an Intersection over Union (IoU) score of 0.95. This paper demonstrates the potential of 3D modeling and AR to revolutionize online shopping by offering realistic virtual interactions, with applicability across broader fashion categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14462v4">LUDVIG: Learning-free Uplifting of 2D Visual features to Gaussian Splatting scenes</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 Project page: https://juliettemarrie.github.io/ludvig
    </div>
    <details class="paper-abstract">
      We address the problem of extending the capabilities of vision foundation models such as DINO, SAM, and CLIP, to 3D tasks. Specifically, we introduce a novel method to uplift 2D image features into Gaussian Splatting representations of 3D scenes. Unlike traditional approaches that rely on minimizing a reconstruction loss, our method employs a simpler and more efficient feature aggregation technique, augmented by a graph diffusion mechanism. Graph diffusion refines 3D features, such as coarse segmentation masks, by leveraging 3D geometry and pairwise similarities induced by DINOv2. Our approach achieves performance comparable to the state of the art on multiple downstream tasks while delivering significant speed-ups. Notably, we obtain competitive segmentation results using generic DINOv2 features, despite DINOv2 not being trained on millions of annotated segmentation masks like SAM. When applied to CLIP features, our method demonstrates strong performance in open-vocabulary object localization tasks, highlighting the versatility of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17085v1">Evaluating CrowdSplat: Perceived Level of Detail for Gaussian Crowds</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Efficient and realistic crowd rendering is an important element of many real-time graphics applications such as Virtual Reality (VR) and games. To this end, Levels of Detail (LOD) avatar representations such as polygonal meshes, image-based impostors, and point clouds have been proposed and evaluated. More recently, 3D Gaussian Splatting has been explored as a potential method for real-time crowd rendering. In this paper, we present a two-alternative forced choice (2AFC) experiment that aims to determine the perceived quality of 3D Gaussian avatars. Three factors were explored: Motion, LOD (i.e., #Gaussians), and the avatar height in Pixels (corresponding to the viewing distance). Participants viewed pairs of animated 3D Gaussian avatars and were tasked with choosing the most detailed one. Our findings can inform the optimization of LOD strategies in Gaussian-based crowd rendering, thereby helping to achieve efficient rendering while maintaining visual quality in real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16764v1">DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-28
      | 💬 Accepted to ICLR 2025; Project page: https://chenguolin.github.io/projects/DiffSplat
    </div>
    <details class="paper-abstract">
      Recent advancements in 3D content generation from text or a single image struggle with limited high-quality 3D datasets and inconsistency from 2D multi-view generation. We introduce DiffSplat, a novel 3D generative framework that natively generates 3D Gaussian splats by taming large-scale text-to-image diffusion models. It differs from previous 3D generative models by effectively utilizing web-scale 2D priors while maintaining 3D consistency in a unified model. To bootstrap the training, a lightweight reconstruction model is proposed to instantly produce multi-view Gaussian splat grids for scalable dataset curation. In conjunction with the regular diffusion loss on these grids, a 3D rendering loss is introduced to facilitate 3D coherence across arbitrary views. The compatibility with image diffusion models enables seamless adaptions of numerous techniques for image generation to the 3D realm. Extensive experiments reveal the superiority of DiffSplat in text- and image-conditioned generation tasks and downstream applications. Thorough ablation studies validate the efficacy of each critical design choice and provide insights into the underlying mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.15619v1">GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-28
    </div>
    <details class="paper-abstract">
      Effective image tokenization is crucial for both multi-modal understanding and generation tasks due to the necessity of the alignment with discrete text data. To this end, existing approaches utilize vector quantization (VQ) to project pixels onto a discrete codebook and reconstruct images from the discrete representation. However, compared with the continuous latent space, the limited discrete codebook space significantly restrict the representational ability of these image tokenizers. In this paper, we propose GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting as a solution. We first represent the encoded samples as multiple flexible featured 2D Gaussians characterized by positions, rotation angles, scaling factors, and feature coefficients. We adopt the standard quantization for the Gaussian features and then concatenate the quantization results with the other intrinsic Gaussian parameters before the corresponding splatting operation and the subsequent decoding module. In general, GaussianToken integrates the local influence of 2D Gaussian distribution into the discrete space and thus enhances the representation capability of the image tokenizer. Competitive reconstruction performances on CIFAR, Mini-ImageNet, and ImageNet-1K demonstrate the effectiveness of our framework. Our code is available at: https://github.com/ChrisDong-THU/GaussianToken.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05731v2">PEP-GS: Perceptually-Enhanced Precise Structured 3D Gaussians for View-Adaptive Rendering</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3D-GS) has achieved significant success in real-time, high-quality 3D scene rendering. However, it faces several challenges, including Gaussian redundancy, limited ability to capture view-dependent effects, and difficulties in handling complex lighting and specular reflections. Additionally, methods that use spherical harmonics for color representation often struggle to effectively capture specular highlights and anisotropic components, especially when modeling view-dependent colors under complex lighting conditions, leading to insufficient contrast and unnatural color saturation. To address these limitations, we introduce PEP-GS, a perceptually-enhanced framework that dynamically predicts Gaussian attributes, including opacity, color, and covariance. We replace traditional spherical harmonics with a Hierarchical Granular-Structural Attention mechanism, which enables more accurate modeling of complex view-dependent color effects and specular highlights. By employing a stable and interpretable framework for opacity and covariance estimation, PEP-GS avoids the removal of essential Gaussians prematurely, ensuring a more accurate scene representation. Furthermore, perceptual optimization is applied to the final rendered images, enhancing perceptual consistency across different views and ensuring high-quality renderings with improved texture fidelity and fine-scale detail preservation. Experimental results demonstrate that PEP-GS outperforms state-of-the-art methods, particularly in challenging scenarios involving view-dependent effects, specular reflections, and fine-scale details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13975v2">3DGS$^2$: Near Second-order Converging 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a mainstream solution for novel view synthesis and 3D reconstruction. By explicitly encoding a 3D scene using a collection of Gaussian kernels, 3DGS achieves high-quality rendering with superior efficiency. As a learning-based approach, 3DGS training has been dealt with the standard stochastic gradient descent (SGD) method, which offers at most linear convergence. Consequently, training often requires tens of minutes, even with GPU acceleration. This paper introduces a (near) second-order convergent training algorithm for 3DGS, leveraging its unique properties. Our approach is inspired by two key observations. First, the attributes of a Gaussian kernel contribute independently to the image-space loss, which endorses isolated and local optimization algorithms. We exploit this by splitting the optimization at the level of individual kernel attributes, analytically constructing small-size Newton systems for each parameter group, and efficiently solving these systems on GPU threads. This achieves Newton-like convergence per training image without relying on the global Hessian. Second, kernels exhibit sparse and structured coupling across input images. This property allows us to effectively utilize spatial information to mitigate overshoot during stochastic training. Our method converges an order faster than standard GPU-based 3DGS training, requiring over $10\times$ fewer iterations while maintaining or surpassing the quality of the compared with the SGD-based 3DGS reconstructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01003v2">EasySplat: View-Adaptive Learning makes 3D Gaussian Splatting Easy</a></div>
    <div class="paper-meta">
      📅 2025-01-27
      | 💬 6 pages, 5figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) techniques have achieved satisfactory 3D scene representation. Despite their impressive performance, they confront challenges due to the limitation of structure-from-motion (SfM) methods on acquiring accurate scene initialization, or the inefficiency of densification strategy. In this paper, we introduce a novel framework EasySplat to achieve high-quality 3DGS modeling. Instead of using SfM for scene initialization, we employ a novel method to release the power of large-scale pointmap approaches. Specifically, we propose an efficient grouping strategy based on view similarity, and use robust pointmap priors to obtain high-quality point clouds and camera poses for 3D scene initialization. After obtaining a reliable scene structure, we propose a novel densification approach that adaptively splits Gaussian primitives based on the average shape of neighboring Gaussian ellipsoids, utilizing KNN scheme. In this way, the proposed method tackles the limitation on initialization and optimization, leading to an efficient and accurate 3DGS modeling. Extensive experiments demonstrate that EasySplat outperforms the current state-of-the-art (SOTA) in handling novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18630v1">Deformable Beta Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-27
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has advanced radiance field reconstruction by enabling real-time rendering. However, its reliance on Gaussian kernels for geometry and low-order Spherical Harmonics (SH) for color encoding limits its ability to capture complex geometries and diverse colors. We introduce Deformable Beta Splatting (DBS), a deformable and compact approach that enhances both geometry and color representation. DBS replaces Gaussian kernels with deformable Beta Kernels, which offer bounded support and adaptive frequency control to capture fine geometric details with higher fidelity while achieving better memory efficiency. In addition, we extended the Beta Kernel to color encoding, which facilitates improved representation of diffuse and specular components, yielding superior results compared to SH-based methods. Furthermore, Unlike prior densification techniques that depend on Gaussian properties, we mathematically prove that adjusting regularized opacity alone ensures distribution-preserved Markov chain Monte Carlo (MCMC), independent of the splatting kernel type. Experimental results demonstrate that DBS achieves state-of-the-art visual quality while utilizing only 45% of the parameters and rendering 1.5x faster than 3DGS-based methods. Notably, for the first time, splatting-based methods outperform state-of-the-art Neural Radiance Fields, highlighting the superior performance and efficiency of DBS for real-time radiance field rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15619v1">GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Effective image tokenization is crucial for both multi-modal understanding and generation tasks due to the necessity of the alignment with discrete text data. To this end, existing approaches utilize vector quantization (VQ) to project pixels onto a discrete codebook and reconstruct images from the discrete representation. However, compared with the continuous latent space, the limited discrete codebook space significantly restrict the representational ability of these image tokenizers. In this paper, we propose GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting as a solution. We first represent the encoded samples as multiple flexible featured 2D Gaussians characterized by positions, rotation angles, scaling factors, and feature coefficients. We adopt the standard quantization for the Gaussian features and then concatenate the quantization results with the other intrinsic Gaussian parameters before the corresponding splatting operation and the subsequent decoding module. In general, GaussianToken integrates the local influence of 2D Gaussian distribution into the discrete space and thus enhances the representation capability of the image tokenizer. Competitive reconstruction performances on CIFAR, Mini-ImageNet, and ImageNet-1K demonstrate the effectiveness of our framework. Our code is available at: https://github.com/ChrisDong-THU/GaussianToken.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00814v2">VR-Doh: Hands-on 3D Modeling in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      We introduce VR-Doh, a hands-on 3D modeling system that enables intuitive creation and manipulation of elastoplastic objects in Virtual Reality (VR). By customizing the Material Point Method (MPM) for real-time simulation of hand-induced large deformations and enhancing 3D Gaussian Splatting for seamless rendering, VR-Doh provides an interactive and immersive 3D modeling experience. Users can naturally sculpt, deform, and edit objects through both contact- and gesture-based hand-object interactions. To achieve real-time performance, our system incorporates localized simulation techniques, particle-level collision handling, and the decoupling of physical and appearance representations, ensuring smooth and responsive interactions. VR-Doh supports both object creation and editing, enabling diverse modeling tasks such as designing food items, characters, and interlocking structures, all resulting in simulation-ready assets. User studies with both novice and experienced participants highlights the system's intuitive design, immersive feedback, and creative potential. Compared to existing geometric modeling tools, VR-Doh offers enhanced accessibility and natural interaction, making it a powerful tool for creative exploration in VR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.15619v1">GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-26
    </div>
    <details class="paper-abstract">
      Effective image tokenization is crucial for both multi-modal understanding and generation tasks due to the necessity of the alignment with discrete text data. To this end, existing approaches utilize vector quantization (VQ) to project pixels onto a discrete codebook and reconstruct images from the discrete representation. However, compared with the continuous latent space, the limited discrete codebook space significantly restrict the representational ability of these image tokenizers. In this paper, we propose GaussianToken: An Effective Image Tokenizer with 2D Gaussian Splatting as a solution. We first represent the encoded samples as multiple flexible featured 2D Gaussians characterized by positions, rotation angles, scaling factors, and feature coefficients. We adopt the standard quantization for the Gaussian features and then concatenate the quantization results with the other intrinsic Gaussian parameters before the corresponding splatting operation and the subsequent decoding module. In general, GaussianToken integrates the local influence of 2D Gaussian distribution into the discrete space and thus enhances the representation capability of the image tokenizer. Competitive reconstruction performances on CIFAR, Mini-ImageNet, and ImageNet-1K demonstrate the effectiveness of our framework. Our code is available at: https://github.com/ChrisDong-THU/GaussianToken.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15096v1">Towards Better Robustness: Progressively Joint Pose-3DGS Learning for Arbitrarily Long Videos</a></div>
    <div class="paper-meta">
      📅 2025-01-25
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a powerful representation due to its efficiency and high-fidelity rendering. However, 3DGS training requires a known camera pose for each input view, typically obtained by Structure-from-Motion (SfM) pipelines. Pioneering works have attempted to relax this restriction but still face difficulties when handling long sequences with complex camera trajectories. In this work, we propose Rob-GS, a robust framework to progressively estimate camera poses and optimize 3DGS for arbitrarily long video sequences. Leveraging the inherent continuity of videos, we design an adjacent pose tracking method to ensure stable pose estimation between consecutive frames. To handle arbitrarily long inputs, we adopt a "divide and conquer" scheme that adaptively splits the video sequence into several segments and optimizes them separately. Extensive experiments on the Tanks and Temples dataset and our collected real-world dataset show that our Rob-GS outperforms the state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15008v1">HuGDiffusion: Generalizable Single-Image Human Rendering via 3D Gaussian Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-01-25
    </div>
    <details class="paper-abstract">
      We present HuGDiffusion, a generalizable 3D Gaussian splatting (3DGS) learning pipeline to achieve novel view synthesis (NVS) of human characters from single-view input images. Existing approaches typically require monocular videos or calibrated multi-view images as inputs, whose applicability could be weakened in real-world scenarios with arbitrary and/or unknown camera poses. In this paper, we aim to generate the set of 3DGS attributes via a diffusion-based framework conditioned on human priors extracted from a single image. Specifically, we begin with carefully integrated human-centric feature extraction procedures to deduce informative conditioning signals. Based on our empirical observations that jointly learning the whole 3DGS attributes is challenging to optimize, we design a multi-stage generation strategy to obtain different types of 3DGS attributes. To facilitate the training process, we investigate constructing proxy ground-truth 3D Gaussian attributes as high-quality attribute-level supervision signals. Through extensive experiments, our HuGDiffusion shows significant performance improvements over the state-of-the-art methods. Our code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14534v1">Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted at ICASSP'25
    </div>
    <details class="paper-abstract">
      Gaussian splatting (GS) for 3D reconstruction has become quite popular due to their fast training, inference speeds and high quality reconstruction. However, GS-based reconstructions generally consist of millions of Gaussians, which makes them hard to use on computationally constrained devices such as smartphones. In this paper, we first propose a principled analysis of advances in efficient GS methods. Then, we propose Trick-GS, which is a careful combination of several strategies including (1) progressive training with resolution, noise and Gaussian scales, (2) learning to prune and mask primitives and SH bands by their significance, and (3) accelerated GS training framework. Trick-GS takes a large step towards resource-constrained GS, where faster run-time, smaller and faster-convergence of models is of paramount concern. Our results on three datasets show that Trick-GS achieves up to 2x faster training, 40x smaller disk size and 2x faster rendering speed compared to vanilla GS, while having comparable accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14319v1">Scalable Benchmarking and Robust Learning for Noise-Free Ego-Motion and 3D Reconstruction from Noisy Video</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted by ICLR 2025; 92 Pages; Project Repo: https://github.com/Xiaohao-Xu/SLAM-under-Perturbation. arXiv admin note: substantial text overlap with arXiv:2406.16850
    </div>
    <details class="paper-abstract">
      We aim to redefine robust ego-motion estimation and photorealistic 3D reconstruction by addressing a critical limitation: the reliance on noise-free data in existing models. While such sanitized conditions simplify evaluation, they fail to capture the unpredictable, noisy complexities of real-world environments. Dynamic motion, sensor imperfections, and synchronization perturbations lead to sharp performance declines when these models are deployed in practice, revealing an urgent need for frameworks that embrace and excel under real-world noise. To bridge this gap, we tackle three core challenges: scalable data generation, comprehensive benchmarking, and model robustness enhancement. First, we introduce a scalable noisy data synthesis pipeline that generates diverse datasets simulating complex motion, sensor imperfections, and synchronization errors. Second, we leverage this pipeline to create Robust-Ego3D, a benchmark rigorously designed to expose noise-induced performance degradation, highlighting the limitations of current learning-based methods in ego-motion accuracy and 3D reconstruction quality. Third, we propose Correspondence-guided Gaussian Splatting (CorrGS), a novel test-time adaptation method that progressively refines an internal clean 3D representation by aligning noisy observations with rendered RGB-D frames from clean 3D map, enhancing geometric alignment and appearance restoration through visual correspondence. Extensive experiments on synthetic and real-world data demonstrate that CorrGS consistently outperforms prior state-of-the-art methods, particularly in scenarios involving rapid motion and dynamic illumination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06814v2">ComPC: Completing a 3D Point Cloud with 2D Diffusion Priors</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      3D point clouds directly collected from objects through sensors are often incomplete due to self-occlusion. Conventional methods for completing these partial point clouds rely on manually organized training sets and are usually limited to object categories seen during training. In this work, we propose a test-time framework for completing partial point clouds across unseen categories without any requirement for training. Leveraging point rendering via Gaussian Splatting, we develop techniques of Partial Gaussian Initialization, Zero-shot Fractal Completion, and Point Cloud Extraction that utilize priors from pre-trained 2D diffusion models to infer missing regions and extract uniform completed point clouds. Experimental results on both synthetic and real-world scanned point clouds demonstrate that our approach outperforms existing methods in completing a variety of objects. Our project page is at \url{https://tianxinhuang.github.io/projects/ComPC/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14277v1">Dense-SfM: Structure from Motion with Dense Consistent Matching</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      We present Dense-SfM, a novel Structure from Motion (SfM) framework designed for dense and accurate 3D reconstruction from multi-view images. Sparse keypoint matching, which traditional SfM methods often rely on, limits both accuracy and point density, especially in texture-less areas. Dense-SfM addresses this limitation by integrating dense matching with a Gaussian Splatting (GS) based track extension which gives more consistent, longer feature tracks. To further improve reconstruction accuracy, Dense-SfM is equipped with a multi-view kernelized matching module leveraging transformer and Gaussian Process architectures, for robust track refinement across multi-views. Evaluations on the ETH3D and Texture-Poor SfM datasets show that Dense-SfM offers significant improvements in accuracy and density over state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14231v1">Micro-macro Wavelet-based Gaussian Splatting for 3D Reconstruction from Unconstrained Images</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 11 pages, 6 figures,accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      3D reconstruction from unconstrained image collections presents substantial challenges due to varying appearances and transient occlusions. In this paper, we introduce Micro-macro Wavelet-based Gaussian Splatting (MW-GS), a novel approach designed to enhance 3D reconstruction by disentangling scene representations into global, refined, and intrinsic components. The proposed method features two key innovations: Micro-macro Projection, which allows Gaussian points to capture details from feature maps across multiple scales with enhanced diversity; and Wavelet-based Sampling, which leverages frequency domain information to refine feature representations and significantly improve the modeling of scene appearances. Additionally, we incorporate a Hierarchical Residual Fusion Network to seamlessly integrate these features. Extensive experiments demonstrate that MW-GS delivers state-of-the-art rendering performance, surpassing existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14147v1">HAMMER: Heterogeneous, Multi-Robot Semantic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting offers expressive scene reconstruction, modeling a broad range of visual, geometric, and semantic information. However, efficient real-time map reconstruction with data streamed from multiple robots and devices remains a challenge. To that end, we propose HAMMER, a server-based collaborative Gaussian Splatting method that leverages widely available ROS communication infrastructure to generate 3D, metric-semantic maps from asynchronous robot data-streams with no prior knowledge of initial robot positions and varying on-device pose estimators. HAMMER consists of (i) a frame alignment module that transforms local SLAM poses and image data into a global frame and requires no prior relative pose knowledge, and (ii) an online module for training semantic 3DGS maps from streaming data. HAMMER handles mixed perception modes, adjusts automatically for variations in image pre-processing among different devices, and distills CLIP semantic codes into the 3D scene for open-vocabulary language queries. In our real-world experiments, HAMMER creates higher-fidelity maps (2x) compared to competing baselines and is useful for downstream tasks, such as semantic goal-conditioned navigation (e.g., ``go to the couch"). Accompanying content available at hammer-project.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.13402v1">VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, map representations based on radiance fields such as 3D Gaussian Splatting and NeRF, which excellent for realistic depiction, have attracted considerable attention, leading to attempts to combine them with SLAM. While these approaches can build highly realistic maps, large-scale SLAM still remains a challenge because they require a large number of Gaussian images for mapping and adjacent images as keyframes for tracking. We propose a novel 3D Gaussian Splatting SLAM method, VIGS SLAM, that utilizes sensor fusion of RGB-D and IMU sensors for large-scale indoor environments. To reduce the computational load of 3DGS-based tracking, we adopt an ICP-based tracking framework that combines IMU preintegration to provide a good initial guess for accurate pose estimation. Our proposed method is the first to propose that Gaussian Splatting-based SLAM can be effectively performed in large-scale environments by integrating IMU sensor measurements. This proposal not only enhances the performance of Gaussian Splatting SLAM beyond room-scale scenarios but also achieves SLAM performance comparable to state-of-the-art methods in large-scale indoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00409v2">3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      In this paper, we present an implicit surface reconstruction method with 3D Gaussian Splatting (3DGS), namely 3DGSR, that allows for accurate 3D reconstruction with intricate details while inheriting the high efficiency and rendering quality of 3DGS. The key insight is incorporating an implicit signed distance field (SDF) within 3D Gaussians to enable them to be aligned and jointly optimized. First, we introduce a differentiable SDF-to-opacity transformation function that converts SDF values into corresponding Gaussians' opacities. This function connects the SDF and 3D Gaussians, allowing for unified optimization and enforcing surface constraints on the 3D Gaussians. During learning, optimizing the 3D Gaussians provides supervisory signals for SDF learning, enabling the reconstruction of intricate details. However, this only provides sparse supervisory signals to the SDF at locations occupied by Gaussians, which is insufficient for learning a continuous SDF. Then, to address this limitation, we incorporate volumetric rendering and align the rendered geometric attributes (depth, normal) with those derived from 3D Gaussians. This consistency regularization introduces supervisory signals to locations not covered by discrete 3D Gaussians, effectively eliminating redundant surfaces outside the Gaussian sampling range. Our extensive experimental results demonstrate that our 3DGSR method enables high-quality 3D surface reconstruction while preserving the efficiency and rendering quality of 3DGS. Besides, our method competes favorably with leading surface reconstruction techniques while offering a more efficient learning process and much better rendering qualities. The code will be available at https://github.com/CVMI-Lab/3DGSR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13558v1">GoDe: Gaussians on Demand for Progressive Level of Detail and Scalable Compression</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting enhances real-time performance in novel view synthesis by representing scenes with mixtures of Gaussians and utilizing differentiable rasterization. However, it typically requires large storage capacity and high VRAM, demanding the design of effective pruning and compression techniques. Existing methods, while effective in some scenarios, struggle with scalability and fail to adapt models based on critical factors such as computing capabilities or bandwidth, requiring to re-train the model under different configurations. In this work, we propose a novel, model-agnostic technique that organizes Gaussians into several hierarchical layers, enabling progressive Level of Detail (LoD) strategy. This method, combined with recent approach of compression of 3DGS, allows a single model to instantly scale across several compression ratios, with minimal to none impact to quality compared to a single non-scalable model and without requiring re-training. We validate our approach on typical datasets and benchmarks, showcasing low distortion and substantial gains in terms of scalability and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13449v1">MultiDreamer3D: Multi-concept 3D Customization with Concept-Aware Diffusion Guidance</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      While single-concept customization has been studied in 3D, multi-concept customization remains largely unexplored. To address this, we propose MultiDreamer3D that can generate coherent multi-concept 3D content in a divide-and-conquer manner. First, we generate 3D bounding boxes using an LLM-based layout controller. Next, a selective point cloud generator creates coarse point clouds for each concept. These point clouds are placed in the 3D bounding boxes and initialized into 3D Gaussian Splatting with concept labels, enabling precise identification of concept attributions in 2D projections. Finally, we refine 3D Gaussians via concept-aware interval score matching, guided by concept-aware diffusion. Our experimental results show that MultiDreamer3D not only ensures object presence and preserves the distinct identities of each concept but also successfully handles complex cases such as property change or interaction. To the best of our knowledge, we are the first to address the multi-concept customization in 3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13417v1">GeomGS: LiDAR-Guided Geometry-Aware Gaussian Splatting for Robot Localization</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Preprint, Under review
    </div>
    <details class="paper-abstract">
      Mapping and localization are crucial problems in robotics and autonomous driving. Recent advances in 3D Gaussian Splatting (3DGS) have enabled precise 3D mapping and scene understanding by rendering photo-realistic images. However, existing 3DGS methods often struggle to accurately reconstruct a 3D map that reflects the actual scale and geometry of the real world, which degrades localization performance. To address these limitations, we propose a novel 3DGS method called Geometry-Aware Gaussian Splatting (GeomGS). This method fully integrates LiDAR data into 3D Gaussian primitives via a probabilistic approach, as opposed to approaches that only use LiDAR as initial points or introduce simple constraints for Gaussian points. To this end, we introduce a Geometric Confidence Score (GCS), which identifies the structural reliability of each Gaussian point. The GCS is optimized simultaneously with Gaussians under probabilistic distance constraints to construct a precise structure. Furthermore, we propose a novel localization method that fully utilizes both the geometric and photometric properties of GeomGS. Our GeomGS demonstrates state-of-the-art geometric and localization performance across several benchmarks, while also improving photometric performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13402v1">VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, map representations based on radiance fields such as 3D Gaussian Splatting and NeRF, which excellent for realistic depiction, have attracted considerable attention, leading to attempts to combine them with SLAM. While these approaches can build highly realistic maps, large-scale SLAM still remains a challenge because they require a large number of Gaussian images for mapping and adjacent images as keyframes for tracking. We propose a novel 3D Gaussian Splatting SLAM method, VIGS SLAM, that utilizes sensor fusion of RGB-D and IMU sensors for large-scale indoor environments. To reduce the computational load of 3DGS-based tracking, we adopt an ICP-based tracking framework that combines IMU preintegration to provide a good initial guess for accurate pose estimation. Our proposed method is the first to propose that Gaussian Splatting-based SLAM can be effectively performed in large-scale environments by integrating IMU sensor measurements. This proposal not only enhances the performance of Gaussian Splatting SLAM beyond room-scale scenarios but also achieves SLAM performance comparable to state-of-the-art methods in large-scale indoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13335v1">Deblur-Avatar: Animatable Avatars from Motion-Blurred Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      We introduce Deblur-Avatar, a novel framework for modeling high-fidelity, animatable 3D human avatars from motion-blurred monocular video inputs. Motion blur is prevalent in real-world dynamic video capture, especially due to human movements in 3D human avatar modeling. Existing methods either (1) assume sharp image inputs, failing to address the detail loss introduced by motion blur, or (2) mainly consider blur by camera movements, neglecting the human motion blur which is more common in animatable avatars. Our proposed approach integrates a human movement-based motion blur model into 3D Gaussian Splatting (3DGS). By explicitly modeling human motion trajectories during exposure time, we jointly optimize the trajectories and 3D Gaussians to reconstruct sharp, high-quality human avatars. We employ a pose-dependent fusion mechanism to distinguish moving body regions, optimizing both blurred and sharp areas effectively. Extensive experiments on synthetic and real-world datasets demonstrate that Deblur-Avatar significantly outperforms existing methods in rendering quality and quantitative metrics, producing sharp avatar reconstructions and enabling real-time rendering under challenging motion blur conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.12060v2">GSVC: Efficient Video Representation and Compression Through 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      3D Gaussian splats have emerged as a revolutionary, effective, learned representation for static 3D scenes. In this work, we explore using 2D Gaussian splats as a new primitive for representing videos. We propose GSVC, an approach to learning a set of 2D Gaussian splats that can effectively represent and compress video frames. GSVC incorporates the following techniques: (i) To exploit temporal redundancy among adjacent frames, which can speed up training and improve the compression efficiency, we predict the Gaussian splats of a frame based on its previous frame; (ii) To control the trade-offs between file size and quality, we remove Gaussian splats with low contribution to the video quality; (iii) To capture dynamics in videos, we randomly add Gaussian splats to fit content with large motion or newly-appeared objects; (iv) To handle significant changes in the scene, we detect key frames based on loss differences during the learning process. Experiment results show that GSVC achieves good rate-distortion trade-offs, comparable to state-of-the-art video codecs such as AV1 and VVC, and a rendering speed of 1500 fps for a 1920x1080 video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.13402v1">VIGS SLAM: IMU-based Large-Scale 3D Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, map representations based on radiance fields such as 3D Gaussian Splatting and NeRF, which excellent for realistic depiction, have attracted considerable attention, leading to attempts to combine them with SLAM. While these approaches can build highly realistic maps, large-scale SLAM still remains a challenge because they require a large number of Gaussian images for mapping and adjacent images as keyframes for tracking. We propose a novel 3D Gaussian Splatting SLAM method, VIGS SLAM, that utilizes sensor fusion of RGB-D and IMU sensors for large-scale indoor environments. To reduce the computational load of 3DGS-based tracking, we adopt an ICP-based tracking framework that combines IMU preintegration to provide a good initial guess for accurate pose estimation. Our proposed method is the first to propose that Gaussian Splatting-based SLAM can be effectively performed in large-scale environments by integrating IMU sensor measurements. This proposal not only enhances the performance of Gaussian Splatting SLAM beyond room-scale scenarios but also achieves SLAM performance comparable to state-of-the-art methods in large-scale indoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13045v1">Sketch and Patch: Efficient 3D Gaussian Representation for Man-Made Scenes</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising representation for photorealistic rendering of 3D scenes. However, its high storage requirements pose significant challenges for practical applications. We observe that Gaussians exhibit distinct roles and characteristics that are analogous to traditional artistic techniques -- Like how artists first sketch outlines before filling in broader areas with color, some Gaussians capture high-frequency features like edges and contours; While other Gaussians represent broader, smoother regions, that are analogous to broader brush strokes that add volume and depth to a painting. Based on this observation, we propose a novel hybrid representation that categorizes Gaussians into (i) Sketch Gaussians, which define scene boundaries, and (ii) Patch Gaussians, which cover smooth regions. Sketch Gaussians are efficiently encoded using parametric models, leveraging their geometric coherence, while Patch Gaussians undergo optimized pruning, retraining, and vector quantization to maintain volumetric consistency and storage efficiency. Our comprehensive evaluation across diverse indoor and outdoor scenes demonstrates that this structure-aware approach achieves up to 32.62% improvement in PSNR, 19.12% in SSIM, and 45.41% in LPIPS at equivalent model sizes, and correspondingly, for an indoor scene, our model maintains the visual quality with 2.3% of the original model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12060v2">GSVC: Efficient Video Representation and Compression Through 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      3D Gaussian splats have emerged as a revolutionary, effective, learned representation for static 3D scenes. In this work, we explore using 2D Gaussian splats as a new primitive for representing videos. We propose GSVC, an approach to learning a set of 2D Gaussian splats that can effectively represent and compress video frames. GSVC incorporates the following techniques: (i) To exploit temporal redundancy among adjacent frames, which can speed up training and improve the compression efficiency, we predict the Gaussian splats of a frame based on its previous frame; (ii) To control the trade-offs between file size and quality, we remove Gaussian splats with low contribution to the video quality; (iii) To capture dynamics in videos, we randomly add Gaussian splats to fit content with large motion or newly-appeared objects; (iv) To handle significant changes in the scene, we detect key frames based on loss differences during the learning process. Experiment results show that GSVC achieves good rate-distortion trade-offs, comparable to state-of-the-art video codecs such as AV1 and VVC, and a rendering speed of 1500 fps for a 1920x1080 video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13971v1">GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      LiDAR novel view synthesis (NVS) has emerged as a novel task within LiDAR simulation, offering valuable simulated point cloud data from novel viewpoints to aid in autonomous driving systems. However, existing LiDAR NVS methods typically rely on neural radiance fields (NeRF) as their 3D representation, which incurs significant computational costs in both training and rendering. Moreover, NeRF and its variants are designed for symmetrical scenes, making them ill-suited for driving scenarios. To address these challenges, we propose GS-LiDAR, a novel framework for generating realistic LiDAR point clouds with panoramic Gaussian splatting. Our approach employs 2D Gaussian primitives with periodic vibration properties, allowing for precise geometric reconstruction of both static and dynamic elements in driving scenarios. We further introduce a novel panoramic rendering technique with explicit ray-splat intersection, guided by panoramic LiDAR supervision. By incorporating intensity and ray-drop spherical harmonic (SH) coefficients into the Gaussian primitives, we enhance the realism of the rendered point clouds. Extensive experiments on KITTI-360 and nuScenes demonstrate the superiority of our method in terms of quantitative metrics, visual quality, as well as training and rendering efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12255v2">HAC++: Towards 100X Compression of 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 Project Page: https://yihangchen-ee.github.io/project_hac++/ Code: https://github.com/YihangChen-ee/HAC-plus. This paper is a journal extension of HAC at arXiv:2403.14530 (ECCV 2024)
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a promising framework for novel view synthesis, boasting rapid rendering speed with high fidelity. However, the substantial Gaussians and their associated attributes necessitate effective compression techniques. Nevertheless, the sparse and unorganized nature of the point cloud of Gaussians (or anchors in our paper) presents challenges for compression. To achieve a compact size, we propose HAC++, which leverages the relationships between unorganized anchors and a structured hash grid, utilizing their mutual information for context modeling. Additionally, HAC++ captures intra-anchor contextual relationships to further enhance compression performance. To facilitate entropy coding, we utilize Gaussian distributions to precisely estimate the probability of each quantized attribute, where an adaptive quantization module is proposed to enable high-precision quantization of these attributes for improved fidelity restoration. Moreover, we incorporate an adaptive masking strategy to eliminate invalid Gaussians and anchors. Overall, HAC++ achieves a remarkable size reduction of over 100X compared to vanilla 3DGS when averaged on all datasets, while simultaneously improving fidelity. It also delivers more than 20X size reduction compared to Scaffold-GS. Our code is available at https://github.com/YihangChen-ee/HAC-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03378v2">Volumetrically Consistent 3D Gaussian Rasterization</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has enabled photorealistic view synthesis at high inference speeds. However, its splatting-based rendering model makes several approximations to the rendering equation, reducing physical accuracy. We show that splatting and its approximations are unnecessary, even within a rasterizer; we instead volumetrically integrate 3D Gaussians directly to compute the transmittance across them analytically. We use this analytic transmittance to derive more physically-accurate alpha values than 3DGS, which can directly be used within their framework. The result is a method that more closely follows the volume rendering equation (similar to ray-tracing) while enjoying the speed benefits of rasterization. Our method represents opaque surfaces with higher accuracy and fewer points than 3DGS. This enables it to outperform 3DGS for view synthesis (measured in SSIM and LPIPS). Being volumetrically consistent also enables our method to work out of the box for tomography. We match the state-of-the-art 3DGS-based tomography method with fewer points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.10788v1">Decoupling Appearance Variations with 3D Consistent Features in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 Accepted to AAAI 2025. Project website: https://davi-gaussian.github.io
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a prominent 3D representation in novel view synthesis, but it still suffers from appearance variations, which are caused by various factors, such as modern camera ISPs, different time of day, weather conditions, and local light changes. These variations can lead to floaters and color distortions in the rendered images/videos. Recent appearance modeling approaches in Gaussian Splatting are either tightly coupled with the rendering process, hindering real-time rendering, or they only account for mild global variations, performing poorly in scenes with local light changes. In this paper, we propose DAVIGS, a method that decouples appearance variations in a plug-and-play and efficient manner. By transforming the rendering results at the image level instead of the Gaussian level, our approach can model appearance variations with minimal optimization time and memory overhead. Furthermore, our method gathers appearance-related information in 3D space to transform the rendered images, thus building 3D consistency across views implicitly. We validate our method on several appearance-variant scenes, and demonstrate that it achieves state-of-the-art rendering quality with minimal training time and memory usage, without compromising rendering speeds. Additionally, it provides performance improvements for different Gaussian Splatting baselines in a plug-and-play manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.12060v2">GSVC: Efficient Video Representation and Compression Through 2D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      3D Gaussian splats have emerged as a revolutionary, effective, learned representation for static 3D scenes. In this work, we explore using 2D Gaussian splats as a new primitive for representing videos. We propose GSVC, an approach to learning a set of 2D Gaussian splats that can effectively represent and compress video frames. GSVC incorporates the following techniques: (i) To exploit temporal redundancy among adjacent frames, which can speed up training and improve the compression efficiency, we predict the Gaussian splats of a frame based on its previous frame; (ii) To control the trade-offs between file size and quality, we remove Gaussian splats with low contribution to the video quality; (iii) To capture dynamics in videos, we randomly add Gaussian splats to fit content with large motion or newly-appeared objects; (iv) To handle significant changes in the scene, we detect key frames based on loss differences during the learning process. Experiment results show that GSVC achieves good rate-distortion trade-offs, comparable to state-of-the-art video codecs such as AV1 and VVC, and a rendering speed of 1500 fps for a 1920x1080 video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12369v1">DARB-Splatting: Generalizing Splatting with Decaying Anisotropic Radial Basis Functions</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Link to the project page: https://randomnerds.github.io/darbs.github.io/
    </div>
    <details class="paper-abstract">
      Splatting-based 3D reconstruction methods have gained popularity with the advent of 3D Gaussian Splatting, efficiently synthesizing high-quality novel views. These methods commonly resort to using exponential family functions, such as the Gaussian function, as reconstruction kernels due to their anisotropic nature, ease of projection, and differentiability in rasterization. However, the field remains restricted to variations within the exponential family, leaving generalized reconstruction kernels largely underexplored, partly due to the lack of easy integrability in 3D to 2D projections. In this light, we show that a class of decaying anisotropic radial basis functions (DARBFs), which are non-negative functions of the Mahalanobis distance, supports splatting by approximating the Gaussian function's closed-form integration advantage. With this fresh perspective, we demonstrate up to 34% faster convergence during training and a 15% reduction in memory consumption across various DARB reconstruction kernels, while maintaining comparable PSNR, SSIM, and LPIPS results. We will make the code available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06714v2">F3D-Gaus: Feed-forward 3D-aware Generation on ImageNet with Cycle-Consistent Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Project Page: https://w-ted.github.io/publications/F3D-Gaus
    </div>
    <details class="paper-abstract">
      This paper tackles the problem of generalizable 3D-aware generation from monocular datasets, e.g., ImageNet. The key challenge of this task is learning a robust 3D-aware representation without multi-view or dynamic data, while ensuring consistent texture and geometry across different viewpoints. Although some baseline methods are capable of 3D-aware generation, the quality of the generated images still lags behind state-of-the-art 2D generation approaches, which excel in producing high-quality, detailed images. To address this severe limitation, we propose a novel feed-forward pipeline based on pixel-aligned Gaussian Splatting, coined as F3D-Gaus, which can produce more realistic and reliable 3D renderings from monocular inputs. In addition, we introduce a self-supervised cycle-consistent constraint to enforce cross-view consistency in the learned 3D representation. This training strategy naturally allows aggregation of multiple aligned Gaussian primitives and significantly alleviates the interpolation limitations inherent in single-view pixel-aligned Gaussian Splatting. Furthermore, we incorporate video model priors to perform geometry-aware refinement, enhancing the generation of fine details in wide-viewpoint scenarios and improving the model's capability to capture intricate 3D textures. Extensive experiments demonstrate that our approach not only achieves high-quality, multi-view consistent 3D-aware generation from monocular datasets, but also significantly improves training and inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03659v4">DehazeGS: Seeing Through Fog with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 9 pages,4 figures. visualizations are available at https://dehazegs.github.io/
    </div>
    <details class="paper-abstract">
      Current novel view synthesis tasks primarily rely on high-quality and clear images. However, in foggy scenes, scattering and attenuation can significantly degrade the reconstruction and rendering quality. Although NeRF-based dehazing reconstruction algorithms have been developed, their use of deep fully connected neural networks and per-ray sampling strategies leads to high computational costs. Moreover, NeRF's implicit representation struggles to recover fine details from hazy scenes. In contrast, recent advancements in 3D Gaussian Splatting achieve high-quality 3D scene reconstruction by explicitly modeling point clouds into 3D Gaussians. In this paper, we propose leveraging the explicit Gaussian representation to explain the foggy image formation process through a physically accurate forward rendering process. We introduce DehazeGS, a method capable of decomposing and rendering a fog-free background from participating media using only muti-view foggy images as input. We model the transmission within each Gaussian distribution to simulate the formation of fog. During this process, we jointly learn the atmospheric light and scattering coefficient while optimizing the Gaussian representation of the hazy scene. In the inference stage, we eliminate the effects of scattering and attenuation on the Gaussians and directly project them onto a 2D plane to obtain a clear view. Experiments on both synthetic and real-world foggy datasets demonstrate that DehazeGS achieves state-of-the-art performance in terms of both rendering quality and computational efficiency. visualizations are available at https://dehazegs.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11508v1">See In Detail: Enhancing Sparse-view 3D Gaussian Splatting with Local Depth and Semantic Regularization</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 5 pages, 5 figures, has been accepted by the ICASSP 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has shown remarkable performance in novel view synthesis. However, its rendering quality deteriorates with sparse inphut views, leading to distorted content and reduced details. This limitation hinders its practical application. To address this issue, we propose a sparse-view 3DGS method. Given the inherently ill-posed nature of sparse-view rendering, incorporating prior information is crucial. We propose a semantic regularization technique, using features extracted from the pretrained DINO-ViT model, to ensure multi-view semantic consistency. Additionally, we propose local depth regularization, which constrains depth values to improve generalization on unseen views. Our method outperforms state-of-the-art novel view synthesis approaches, achieving up to 0.4dB improvement in terms of PSNR on the LLFF dataset, with reduced distortion and enhanced visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10283v2">GSTAR: Gaussian Surface Tracking and Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting techniques have enabled efficient photo-realistic rendering of static scenes. Recent works have extended these approaches to support surface reconstruction and tracking. However, tracking dynamic surfaces with 3D Gaussians remains challenging due to complex topology changes, such as surfaces appearing, disappearing, or splitting. To address these challenges, we propose GSTAR, a novel method that achieves photo-realistic rendering, accurate surface reconstruction, and reliable 3D tracking for general dynamic scenes with changing topology. Given multi-view captures as input, GSTAR binds Gaussians to mesh faces to represent dynamic objects. For surfaces with consistent topology, GSTAR maintains the mesh topology and tracks the meshes using Gaussians. In regions where topology changes, GSTAR adaptively unbinds Gaussians from the mesh, enabling accurate registration and the generation of new surfaces based on these optimized Gaussians. Additionally, we introduce a surface-based scene flow method that provides robust initialization for tracking between frames. Experiments demonstrate that our method effectively tracks and reconstructs dynamic surfaces, enabling a range of applications. Our project page with the code release is available at https://eth-ait.github.io/GSTAR/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11102v1">RDG-GS: Relative Depth Guidance with Gaussian Splatting for Real-time Sparse-View 3D Rendering</a></div>
    <div class="paper-meta">
      📅 2025-01-19
      | 💬 24 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Efficiently synthesizing novel views from sparse inputs while maintaining accuracy remains a critical challenge in 3D reconstruction. While advanced techniques like radiance fields and 3D Gaussian Splatting achieve rendering quality and impressive efficiency with dense view inputs, they suffer from significant geometric reconstruction errors when applied to sparse input views. Moreover, although recent methods leverage monocular depth estimation to enhance geometric learning, their dependence on single-view estimated depth often leads to view inconsistency issues across different viewpoints. Consequently, this reliance on absolute depth can introduce inaccuracies in geometric information, ultimately compromising the quality of scene reconstruction with Gaussian splats. In this paper, we present RDG-GS, a novel sparse-view 3D rendering framework with Relative Depth Guidance based on 3D Gaussian Splatting. The core innovation lies in utilizing relative depth guidance to refine the Gaussian field, steering it towards view-consistent spatial geometric representations, thereby enabling the reconstruction of accurate geometric structures and capturing intricate textures. First, we devise refined depth priors to rectify the coarse estimated depth and insert global and fine-grained scene information to regular Gaussians. Building on this, to address spatial geometric inaccuracies from absolute depth, we propose relative depth guidance by optimizing the similarity between spatially correlated patches of depth and images. Additionally, we also directly deal with the sparse areas challenging to converge by the adaptive sampling for quick densification. Across extensive experiments on Mip-NeRF360, LLFF, DTU, and Blender, RDG-GS demonstrates state-of-the-art rendering quality and efficiency, making a significant advancement for real-world application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17857v4">SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-01-19
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has emerged as an alternative 3D representation for novel view synthesis, benefiting from its high-quality rendering results and real-time rendering speed. However, the 3D Gaussians learned by 3D-GS have ambiguous structures without any geometry constraints. This inherent issue in 3D-GS leads to a rough boundary when segmenting individual objects. To remedy these problems, we propose SAGD, a conceptually simple yet effective boundary-enhanced segmentation pipeline for 3D-GS to improve segmentation accuracy while preserving segmentation speed. Specifically, we introduce a Gaussian Decomposition scheme, which ingeniously utilizes the special structure of 3D Gaussian, finds out, and then decomposes the boundary Gaussians. Moreover, to achieve fast interactive 3D segmentation, we introduce a novel training-free pipeline by lifting a 2D foundation model to 3D-GS. Extensive experiments demonstrate that our approach achieves high-quality 3D segmentation without rough boundary issues, which can be easily applied to other scene editing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10788v1">Decoupling Appearance Variations with 3D Consistent Features in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-18
      | 💬 Accepted to AAAI 2025. Project website: https://davi-gaussian.github.io
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a prominent 3D representation in novel view synthesis, but it still suffers from appearance variations, which are caused by various factors, such as modern camera ISPs, different time of day, weather conditions, and local light changes. These variations can lead to floaters and color distortions in the rendered images/videos. Recent appearance modeling approaches in Gaussian Splatting are either tightly coupled with the rendering process, hindering real-time rendering, or they only account for mild global variations, performing poorly in scenes with local light changes. In this paper, we propose DAVIGS, a method that decouples appearance variations in a plug-and-play and efficient manner. By transforming the rendering results at the image level instead of the Gaussian level, our approach can model appearance variations with minimal optimization time and memory overhead. Furthermore, our method gathers appearance-related information in 3D space to transform the rendered images, thus building 3D consistency across views implicitly. We validate our method on several appearance-variant scenes, and demonstrate that it achieves state-of-the-art rendering quality with minimal training time and memory usage, without compromising rendering speeds. Additionally, it provides performance improvements for different Gaussian Splatting baselines in a plug-and-play manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03706v2">3DGS-CD: 3D Gaussian Splatting-based Change Detection for Physical Object Rearrangement</a></div>
    <div class="paper-meta">
      📅 2025-01-18
    </div>
    <details class="paper-abstract">
      We present 3DGS-CD, the first 3D Gaussian Splatting (3DGS)-based method for detecting physical object rearrangements in 3D scenes. Our approach estimates 3D object-level changes by comparing two sets of unaligned images taken at different times. Leveraging 3DGS's novel view rendering and EfficientSAM's zero-shot segmentation capabilities, we detect 2D object-level changes, which are then associated and fused across views to estimate 3D change masks and object transformations. Our method can accurately identify changes in cluttered environments using sparse (as few as one) post-change images within as little as 18s. It does not rely on depth input, user instructions, pre-defined object classes, or object models -- An object is recognized simply if it has been re-arranged. Our approach is evaluated on both public and self-collected real-world datasets, achieving up to 14% higher accuracy and three orders of magnitude faster performance compared to the state-of-the-art radiance-field-based change detection method. This significant performance boost enables a broad range of downstream applications, where we highlight three key use cases: object reconstruction, robot workspace reset, and 3DGS model update. Our code and data will be made available at https://github.com/520xyxyzq/3DGS-CD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.10788v1">Decoupling Appearance Variations with 3D Consistent Features in Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-18
      | 💬 Accepted to AAAI 2025. Project website: https://davi-gaussian.github.io
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a prominent 3D representation in novel view synthesis, but it still suffers from appearance variations, which are caused by various factors, such as modern camera ISPs, different time of day, weather conditions, and local light changes. These variations can lead to floaters and color distortions in the rendered images/videos. Recent appearance modeling approaches in Gaussian Splatting are either tightly coupled with the rendering process, hindering real-time rendering, or they only account for mild global variations, performing poorly in scenes with local light changes. In this paper, we propose DAVIGS, a method that decouples appearance variations in a plug-and-play and efficient manner. By transforming the rendering results at the image level instead of the Gaussian level, our approach can model appearance variations with minimal optimization time and memory overhead. Furthermore, our method gathers appearance-related information in 3D space to transform the rendered images, thus building 3D consistency across views implicitly. We validate our method on several appearance-variant scenes, and demonstrate that it achieves state-of-the-art rendering quality with minimal training time and memory usage, without compromising rendering speeds. Additionally, it provides performance improvements for different Gaussian Splatting baselines in a plug-and-play manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11396v2">Beyond Uncertainty: Risk-Aware Active View Acquisition for Safe Robot Navigation and 3D Scene Understanding with FisherRF</a></div>
    <div class="paper-meta">
      📅 2025-01-17
    </div>
    <details class="paper-abstract">
      The active view acquisition problem has been extensively studied in the context of robot navigation using NeRF and 3D Gaussian Splatting. To enhance scene reconstruction efficiency and ensure robot safety, we propose the Risk-aware Environment Masking (RaEM) framework. RaEM leverages coherent risk measures to dynamically prioritize safety-critical regions of the unknown environment, guiding active view acquisition algorithms toward identifying the next-best-view (NBV). Integrated with FisherRF, which selects the NBV by maximizing expected information gain, our framework achieves a dual objective: improving robot safety and increasing efficiency in risk-aware 3D scene reconstruction and understanding. Extensive high-fidelity experiments validate the effectiveness of our approach, demonstrating its ability to establish a robust and safety-focused framework for active robot exploration and 3D scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09302v1">Creating Virtual Environments with 3D Gaussian Splatting: A Comparative Study</a></div>
    <div class="paper-meta">
      📅 2025-01-16
      | 💬 IEEE VR 2025 Posters
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently emerged as an innovative and efficient 3D representation technique. While its potential for extended reality (XR) applications is frequently highlighted, its practical effectiveness remains underexplored. In this work, we examine three distinct 3DGS-based approaches for virtual environment (VE) creation, leveraging their unique strengths for efficient and visually compelling scene representation. By conducting a comparable study, we evaluate the feasibility of 3DGS in creating immersive VEs, identify its limitations in XR applications, and discuss future research and development opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08982v1">CityLoc: 6 DoF Localization of Text Descriptions in Large-Scale Scenes with Gaussian Representation</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      Localizing text descriptions in large-scale 3D scenes is inherently an ambiguous task. This nonetheless arises while describing general concepts, e.g. all traffic lights in a city. To facilitate reasoning based on such concepts, text localization in the form of distribution is required. In this paper, we generate the distribution of the camera poses conditioned upon the textual description. To facilitate such generation, we propose a diffusion-based architecture that conditionally diffuses the noisy 6DoF camera poses to their plausible locations. The conditional signals are derived from the text descriptions, using the pre-trained text encoders. The connection between text descriptions and pose distribution is established through pretrained Vision-Language-Model, i.e. CLIP. Furthermore, we demonstrate that the candidate poses for the distribution can be further refined by rendering potential poses using 3D Gaussian splatting, guiding incorrectly posed samples towards locations that better align with the textual description, through visual reasoning. We demonstrate the effectiveness of our method by comparing it with both standard retrieval methods and learning-based approaches. Our proposed method consistently outperforms these baselines across all five large-scale datasets. Our source code and dataset will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10462v1">BloomScene: Lightweight Structured 3D Gaussian Splatting for Crossmodal Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      With the widespread use of virtual reality applications, 3D scene generation has become a new challenging research frontier. 3D scenes have highly complex structures and need to ensure that the output is dense, coherent, and contains all necessary structures. Many current 3D scene generation methods rely on pre-trained text-to-image diffusion models and monocular depth estimators. However, the generated scenes occupy large amounts of storage space and often lack effective regularisation methods, leading to geometric distortions. To this end, we propose BloomScene, a lightweight structured 3D Gaussian splatting for crossmodal scene generation, which creates diverse and high-quality 3D scenes from text or image inputs. Specifically, a crossmodal progressive scene generation framework is proposed to generate coherent scenes utilizing incremental point cloud reconstruction and 3D Gaussian splatting. Additionally, we propose a hierarchical depth prior-based regularization mechanism that utilizes multi-level constraints on depth accuracy and smoothness to enhance the realism and continuity of the generated scenes. Ultimately, we propose a structured context-guided compression mechanism that exploits structured hash grids to model the context of unorganized anchor attributes, which significantly eliminates structural redundancy and reduces storage overhead. Comprehensive experiments across multiple scenes demonstrate the significant potential and advantages of our framework compared with several baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08672v1">GS-LIVO: Real-Time LiDAR, Inertial, and Visual Multi-sensor Fused Odometry with Gaussian Mapping</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      In recent years, 3D Gaussian splatting (3D-GS) has emerged as a novel scene representation approach. However, existing vision-only 3D-GS methods often rely on hand-crafted heuristics for point-cloud densification and face challenges in handling occlusions and high GPU memory and computation consumption. LiDAR-Inertial-Visual (LIV) sensor configuration has demonstrated superior performance in localization and dense mapping by leveraging complementary sensing characteristics: rich texture information from cameras, precise geometric measurements from LiDAR, and high-frequency motion data from IMU. Inspired by this, we propose a novel real-time Gaussian-based simultaneous localization and mapping (SLAM) system. Our map system comprises a global Gaussian map and a sliding window of Gaussians, along with an IESKF-based odometry. The global Gaussian map consists of hash-indexed voxels organized in a recursive octree, effectively covering sparse spatial volumes while adapting to different levels of detail and scales. The Gaussian map is initialized through multi-sensor fusion and optimized with photometric gradients. Our system incrementally maintains a sliding window of Gaussians, significantly reducing GPU computation and memory consumption by only optimizing the map within the sliding window. Moreover, we implement a tightly coupled multi-sensor fusion odometry with an iterative error state Kalman filter (IESKF), leveraging real-time updating and rendering of the Gaussian map. Our system represents the first real-time Gaussian-based SLAM framework deployable on resource-constrained embedded systems, demonstrated on the NVIDIA Jetson Orin NX platform. The framework achieves real-time performance while maintaining robust multi-sensor fusion capabilities. All implementation algorithms, hardware designs, and CAD models will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07015v2">SplatMAP: Online Dense Monocular SLAM with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Achieving high-fidelity 3D reconstruction from monocular video remains challenging due to the inherent limitations of traditional methods like Structure-from-Motion (SfM) and monocular SLAM in accurately capturing scene details. While differentiable rendering techniques such as Neural Radiance Fields (NeRF) address some of these challenges, their high computational costs make them unsuitable for real-time applications. Additionally, existing 3D Gaussian Splatting (3DGS) methods often focus on photometric consistency, neglecting geometric accuracy and failing to exploit SLAM's dynamic depth and pose updates for scene refinement. We propose a framework integrating dense SLAM with 3DGS for real-time, high-fidelity dense reconstruction. Our approach introduces SLAM-Informed Adaptive Densification, which dynamically updates and densifies the Gaussian model by leveraging dense point clouds from SLAM. Additionally, we incorporate Geometry-Guided Optimization, which combines edge-aware geometric constraints and photometric consistency to jointly optimize the appearance and geometry of the 3DGS scene representation, enabling detailed and accurate SLAM mapping reconstruction. Experiments on the Replica and TUM-RGBD datasets demonstrate the effectiveness of our approach, achieving state-of-the-art results among monocular systems. Specifically, our method achieves a PSNR of 36.864, SSIM of 0.985, and LPIPS of 0.040 on Replica, representing improvements of 10.7%, 6.4%, and 49.4%, respectively, over the previous SOTA. On TUM-RGBD, our method outperforms the closest baseline by 10.2%, 6.6%, and 34.7% in the same metrics. These results highlight the potential of our framework in bridging the gap between photometric and geometric dense 3D scene representations, paving the way for practical and efficient monocular dense reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19420v2">RF-3DGS: Wireless Channel Modeling with Radio Radiance Field and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 in submission to IEEE journals
    </div>
    <details class="paper-abstract">
      Precisely modeling radio propagation in complex environments has been a significant challenge, especially with the advent of 5G and beyond networks, where managing massive antenna arrays demands more detailed information. Traditional methods, such as empirical models and ray tracing, often fall short, either due to insufficient details or because of challenges for real-time applications. Inspired by the newly proposed 3D Gaussian Splatting method in the computer vision domain, which outperforms other methods in reconstructing optical radiance fields, we propose RF-3DGS, a novel approach that enables precise site-specific reconstruction of radio radiance fields from sparse samples. RF-3DGS can render radio spatial spectra at arbitrary positions within 2 ms following a brief 3-minute training period, effectively identifying dominant propagation paths. Furthermore, RF-3DGS can provide fine-grained Spatial Channel State Information (Spatial-CSI) of these paths, including the channel gain, the delay, the angle of arrival (AoA), and the angle of departure (AoD). Our experiments, calibrated through real-world measurements, demonstrate that RF-3DGS not only significantly improves reconstruction quality, training efficiency, and rendering speed compared to state-of-the-art methods, but also holds great potential for supporting wireless communication and advanced applications such as Integrated Sensing and Communication (ISAC). Code and dataset will be available at https://github.com/SunLab-UGA/RF-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08370v1">3D Gaussian Splatting with Normal Information for Mesh Extraction and Improved Rendering</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 ICASSP 2025: Workshop on Generative Data Augmentation for Real-World Signal Processing Applications
    </div>
    <details class="paper-abstract">
      Differentiable 3D Gaussian splatting has emerged as an efficient and flexible rendering technique for representing complex scenes from a collection of 2D views and enabling high-quality real-time novel-view synthesis. However, its reliance on photometric losses can lead to imprecisely reconstructed geometry and extracted meshes, especially in regions with high curvature or fine detail. We propose a novel regularization method using the gradients of a signed distance function estimated from the Gaussians, to improve the quality of rendering while also extracting a surface mesh. The regularizing normal supervision facilitates better rendering and mesh reconstruction, which is crucial for downstream applications in video generation, animation, AR-VR and gaming. We demonstrate the effectiveness of our approach on datasets such as Mip-NeRF360, Tanks and Temples, and Deep-Blending. Our method scores higher on photorealism metrics compared to other mesh extracting rendering methods without compromising mesh quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04545v2">Gaussian Eigen Models for Human Heads</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 https://zielon.github.io/gem/
    </div>
    <details class="paper-abstract">
      Current personalized neural head avatars face a trade-off: lightweight models lack detail and realism, while high-quality, animatable avatars require significant computational resources, making them unsuitable for commodity devices. To address this gap, we introduce Gaussian Eigen Models (GEM), which provide high-quality, lightweight, and easily controllable head avatars. GEM utilizes 3D Gaussian primitives for representing the appearance combined with Gaussian splatting for rendering. Building on the success of mesh-based 3D morphable face models (3DMM), we define GEM as an ensemble of linear eigenbases for representing the head appearance of a specific subject. In particular, we construct linear bases to represent the position, scale, rotation, and opacity of the 3D Gaussians. This allows us to efficiently generate Gaussian primitives of a specific head shape by a linear combination of the basis vectors, only requiring a low-dimensional parameter vector that contains the respective coefficients. We propose to construct these linear bases (GEM) by distilling high-quality compute-intense CNN-based Gaussian avatar models that can generate expression-dependent appearance changes like wrinkles. These high-quality models are trained on multi-view videos of a subject and are distilled using a series of principal component analyses. Once we have obtained the bases that represent the animatable appearance space of a specific human, we learn a regressor that takes a single RGB image as input and predicts the low-dimensional parameter vector that corresponds to the shown facial expression. In a series of experiments, we compare GEM's self-reenactment and cross-person reenactment results to state-of-the-art 3D avatar methods, demonstrating GEM's higher visual quality and better generalization to new expressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08286v1">VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      VINGS-Mono is a monocular (inertial) Gaussian Splatting (GS) SLAM framework designed for large scenes. The framework comprises four main components: VIO Front End, 2D Gaussian Map, NVS Loop Closure, and Dynamic Eraser. In the VIO Front End, RGB frames are processed through dense bundle adjustment and uncertainty estimation to extract scene geometry and poses. Based on this output, the mapping module incrementally constructs and maintains a 2D Gaussian map. Key components of the 2D Gaussian Map include a Sample-based Rasterizer, Score Manager, and Pose Refinement, which collectively improve mapping speed and localization accuracy. This enables the SLAM system to handle large-scale urban environments with up to 50 million Gaussian ellipsoids. To ensure global consistency in large-scale scenes, we design a Loop Closure module, which innovatively leverages the Novel View Synthesis (NVS) capabilities of Gaussian Splatting for loop closure detection and correction of the Gaussian map. Additionally, we propose a Dynamic Eraser to address the inevitable presence of dynamic objects in real-world outdoor scenes. Extensive evaluations in indoor and outdoor environments demonstrate that our approach achieves localization performance on par with Visual-Inertial Odometry while surpassing recent GS/NeRF SLAM methods. It also significantly outperforms all existing methods in terms of mapping and rendering quality. Furthermore, we developed a mobile app and verified that our framework can generate high-quality Gaussian maps in real time using only a smartphone camera and a low-frequency IMU sensor. To the best of our knowledge, VINGS-Mono is the first monocular Gaussian SLAM method capable of operating in outdoor environments and supporting kilometer-scale large scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08174v1">Object-Centric 2D Gaussian Splatting: Background Removal and Occlusion-Aware Pruning for Compact Object Models</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 Accepted at ICPRAM 2025 (https://icpram.scitevents.org/Home.aspx)
    </div>
    <details class="paper-abstract">
      Current Gaussian Splatting approaches are effective for reconstructing entire scenes but lack the option to target specific objects, making them computationally expensive and unsuitable for object-specific applications. We propose a novel approach that leverages object masks to enable targeted reconstruction, resulting in object-centric models. Additionally, we introduce an occlusion-aware pruning strategy to minimize the number of Gaussians without compromising quality. Our method reconstructs compact object models, yielding object-centric Gaussian and mesh representations that are up to 96\% smaller and up to 71\% faster to train compared to the baseline while retaining competitive quality. These representations are immediately usable for downstream applications such as appearance editing and physics simulation without additional processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06838v2">Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale Super-Resolution</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Equipped with the continuous representation capability of Multi-Layer Perceptron (MLP), Implicit Neural Representation (INR) has been successfully employed for Arbitrary-scale Super-Resolution (ASR). However, the limited receptive field of the linear layers in MLP restricts the representation capability of INR, while it is computationally expensive to query the MLP numerous times to render each pixel. Recently, Gaussian Splatting (GS) has shown its advantages over INR in both visual quality and rendering speed in 3D tasks, which motivates us to explore whether GS can be employed for the ASR task. However, directly applying GS to ASR is exceptionally challenging because the original GS is an optimization-based method through overfitting each single scene, while in ASR we aim to learn a single model that can generalize to different images and scaling factors. We overcome these challenges by developing two novel techniques. Firstly, to generalize GS for ASR, we elaborately design an architecture to predict the corresponding image-conditioned Gaussians of the input low-resolution image in a feed-forward manner. Secondly, we implement an efficient differentiable 2D GPU/CUDA-based scale-aware rasterization to render super-resolved images by sampling discrete RGB values from the predicted contiguous Gaussians. Via end-to-end training, our optimized network, namely GSASR, can perform ASR for any image and unseen scaling factors. Extensive experiments validate the effectiveness of our proposed method. The project page can be found at \url{https://mt-cly.github.io/GSASR.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.02751v3">Splat-Nav: Safe Real-Time Robot Navigation in Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      We present Splat-Nav, a real-time robot navigation pipeline for Gaussian Splatting (GSplat) scenes, a powerful new 3D scene representation. Splat-Nav consists of two components: 1) Splat-Plan, a safe planning module, and 2) Splat-Loc, a robust vision-based pose estimation module. Splat-Plan builds a safe-by-construction polytope corridor through the map based on mathematically rigorous collision constraints and then constructs a Bézier curve trajectory through this corridor. Splat-Loc provides real-time recursive state estimates given only an RGB feed from an on-board camera, leveraging the point-cloud representation inherent in GSplat scenes. Working together, these modules give robots the ability to recursively re-plan smooth and safe trajectories to goal locations. Goals can be specified with position coordinates, or with language commands by using a semantic GSplat. We demonstrate improved safety compared to point cloud-based methods in extensive simulation experiments. In a total of 126 hardware flights, we demonstrate equivalent safety and speed compared to motion capture and visual odometry, but without a manual frame alignment required by those methods. We show online re-planning at more than 2 Hz and pose estimation at about 25 Hz, an order of magnitude faster than Neural Radiance Field (NeRF)-based navigation methods, thereby enabling real-time navigation. We provide experiment videos on our project page at https://chengine.github.io/splatnav/. Our codebase and ROS nodes can be found at https://github.com/chengine/splatnav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07574v1">UnCommon Objects in 3D</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      We introduce Uncommon Objects in 3D (uCO3D), a new object-centric dataset for 3D deep learning and 3D generative AI. uCO3D is the largest publicly-available collection of high-resolution videos of objects with 3D annotations that ensures full-360$^{\circ}$ coverage. uCO3D is significantly more diverse than MVImgNet and CO3Dv2, covering more than 1,000 object categories. It is also of higher quality, due to extensive quality checks of both the collected videos and the 3D annotations. Similar to analogous datasets, uCO3D contains annotations for 3D camera poses, depth maps and sparse point clouds. In addition, each object is equipped with a caption and a 3D Gaussian Splat reconstruction. We train several large 3D models on MVImgNet, CO3Dv2, and uCO3D and obtain superior results using the latter, showing that uCO3D is better for learning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05379v2">Arc2Avatar: Generating Expressive 3D Avatars from a Single Image via ID Guidance</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 Project Page https://arc2avatar.github.io
    </div>
    <details class="paper-abstract">
      Inspired by the effectiveness of 3D Gaussian Splatting (3DGS) in reconstructing detailed 3D scenes within multi-view setups and the emergence of large 2D human foundation models, we introduce Arc2Avatar, the first SDS-based method utilizing a human face foundation model as guidance with just a single image as input. To achieve that, we extend such a model for diverse-view human head generation by fine-tuning on synthetic data and modifying its conditioning. Our avatars maintain a dense correspondence with a human face mesh template, allowing blendshape-based expression generation. This is achieved through a modified 3DGS approach, connectivity regularizers, and a strategic initialization tailored for our task. Additionally, we propose an optional efficient SDS-based correction step to refine the blendshape expressions, enhancing realism and diversity. Experiments demonstrate that Arc2Avatar achieves state-of-the-art realism and identity preservation, effectively addressing color issues by allowing the use of very low guidance, enabled by our strong identity prior and initialization strategy, without compromising detail. Please visit https://arc2avatar.github.io for more resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07478v1">3DGS-to-PC: Convert a 3D Gaussian Splatting Scene into a Dense Point Cloud or Mesh</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) excels at producing highly detailed 3D reconstructions, but these scenes often require specialised renderers for effective visualisation. In contrast, point clouds are a widely used 3D representation and are compatible with most popular 3D processing software, yet converting 3DGS scenes into point clouds is a complex challenge. In this work we introduce 3DGS-to-PC, a flexible and highly customisable framework that is capable of transforming 3DGS scenes into dense, high-accuracy point clouds. We sample points probabilistically from each Gaussian as a 3D density function. We additionally threshold new points using the Mahalanobis distance to the Gaussian centre, preventing extreme outliers. The result is a point cloud that closely represents the shape encoded into the 3D Gaussian scene. Individual Gaussians use spherical harmonics to adapt colours depending on view, and each point may contribute only subtle colour hints to the resulting rendered scene. To avoid spurious or incorrect colours that do not fit with the final point cloud, we recalculate Gaussian colours via a customised image rendering approach, assigning each Gaussian the colour of the pixel to which it contributes most across all views. 3DGS-to-PC also supports mesh generation through Poisson Surface Reconstruction, applied to points sampled from predicted surface Gaussians. This allows coloured meshes to be generated from 3DGS scenes without the need for re-training. This package is highly customisable and capability of simple integration into existing 3DGS pipelines. 3DGS-to-PC provides a powerful tool for converting 3DGS data into point cloud and surface-based formats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08072v1">Evaluating Human Perception of Novel View Synthesis: Subjective Quality Assessment of Gaussian Splatting and NeRF in Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Gaussian Splatting (GS) and Neural Radiance Fields (NeRF) are two groundbreaking technologies that have revolutionized the field of Novel View Synthesis (NVS), enabling immersive photorealistic rendering and user experiences by synthesizing multiple viewpoints from a set of images of sparse views. The potential applications of NVS, such as high-quality virtual and augmented reality, detailed 3D modeling, and realistic medical organ imaging, underscore the importance of quality assessment of NVS methods from the perspective of human perception. Although some previous studies have explored subjective quality assessments for NVS technology, they still face several challenges, especially in NVS methods selection, scenario coverage, and evaluation methodology. To address these challenges, we conducted two subjective experiments for the quality assessment of NVS technologies containing both GS-based and NeRF-based methods, focusing on dynamic and real-world scenes. This study covers 360{\deg}, front-facing, and single-viewpoint videos while providing a richer and greater number of real scenes. Meanwhile, it's the first time to explore the impact of NVS methods in dynamic scenes with moving objects. The two types of subjective experiments help to fully comprehend the influences of different viewing paths from a human perception perspective and pave the way for future development of full-reference and no-reference quality metrics. In addition, we established a comprehensive benchmark of various state-of-the-art objective metrics on the proposed database, highlighting that existing methods still struggle to accurately capture subjective quality. The results give us some insights into the limitations of existing NVS methods and may promote the development of new NVS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06019v2">HeadGAP: Few-Shot 3D Head Avatar via Generalizable Gaussian Priors</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 Accepted to 3DV 2025. Project page: https://headgap.github.io/
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel 3D head avatar creation approach capable of generalizing from few-shot in-the-wild data with high-fidelity and animatable robustness. Given the underconstrained nature of this problem, incorporating prior knowledge is essential. Therefore, we propose a framework comprising prior learning and avatar creation phases. The prior learning phase leverages 3D head priors derived from a large-scale multi-view dynamic dataset, and the avatar creation phase applies these priors for few-shot personalization. Our approach effectively captures these priors by utilizing a Gaussian Splatting-based auto-decoder network with part-based dynamic modeling. Our method employs identity-shared encoding with personalized latent codes for individual identities to learn the attributes of Gaussian primitives. During the avatar creation phase, we achieve fast head avatar personalization by leveraging inversion and fine-tuning strategies. Extensive experiments demonstrate that our model effectively exploits head priors and successfully generalizes them to few-shot personalization, achieving photo-realistic rendering quality, multi-view consistency, and stable animation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07104v1">RMAvatar: Photorealistic Human Avatar Reconstruction from Monocular Video Based on Rectified Mesh-embedded Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 CVM2025
    </div>
    <details class="paper-abstract">
      We introduce RMAvatar, a novel human avatar representation with Gaussian splatting embedded on mesh to learn clothed avatar from a monocular video. We utilize the explicit mesh geometry to represent motion and shape of a virtual human and implicit appearance rendering with Gaussian Splatting. Our method consists of two main modules: Gaussian initialization module and Gaussian rectification module. We embed Gaussians into triangular faces and control their motion through the mesh, which ensures low-frequency motion and surface deformation of the avatar. Due to the limitations of LBS formula, the human skeleton is hard to control complex non-rigid transformations. We then design a pose-related Gaussian rectification module to learn fine-detailed non-rigid deformations, further improving the realism and expressiveness of the avatar. We conduct extensive experiments on public datasets, RMAvatar shows state-of-the-art performance on both rendering quality and quantitative evaluations. Please see our project page at https://rm-avatar.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06927v1">CULTURE3D: Cultural Landmarks and Terrain Dataset for 3D Applications</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      In this paper, we present a large-scale fine-grained dataset using high-resolution images captured from locations worldwide. Compared to existing datasets, our dataset offers a significantly larger size and includes a higher level of detail, making it uniquely suited for fine-grained 3D applications. Notably, our dataset is built using drone-captured aerial imagery, which provides a more accurate perspective for capturing real-world site layouts and architectural structures. By reconstructing environments with these detailed images, our dataset supports applications such as the COLMAP format for Gaussian Splatting and the Structure-from-Motion (SfM) method. It is compatible with widely-used techniques including SLAM, Multi-View Stereo, and Neural Radiance Fields (NeRF), enabling accurate 3D reconstructions and point clouds. This makes it a benchmark for reconstruction and segmentation tasks. The dataset enables seamless integration with multi-modal data, supporting a range of 3D applications, from architectural reconstruction to virtual tourism. Its flexibility promotes innovation, facilitating breakthroughs in 3D modeling and analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06903v1">Synthetic Prior for Few-Shot Drivable Head Avatar Inversion</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Website https://zielon.github.io/synshot/
    </div>
    <details class="paper-abstract">
      We present SynShot, a novel method for the few-shot inversion of a drivable head avatar based on a synthetic prior. We tackle two major challenges. First, training a controllable 3D generative network requires a large number of diverse sequences, for which pairs of images and high-quality tracked meshes are not always available. Second, state-of-the-art monocular avatar models struggle to generalize to new views and expressions, lacking a strong prior and often overfitting to a specific viewpoint distribution. Inspired by machine learning models trained solely on synthetic data, we propose a method that learns a prior model from a large dataset of synthetic heads with diverse identities, expressions, and viewpoints. With few input images, SynShot fine-tunes the pretrained synthetic prior to bridge the domain gap, modeling a photorealistic head avatar that generalizes to novel expressions and viewpoints. We model the head avatar using 3D Gaussian splatting and a convolutional encoder-decoder that outputs Gaussian parameters in UV texture space. To account for the different modeling complexities over parts of the head (e.g., skin vs hair), we embed the prior with explicit control for upsampling the number of per-part primitives. Compared to state-of-the-art monocular methods that require thousands of real training images, SynShot significantly improves novel view and expression synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06897v1">ActiveGAMER: Active GAussian Mapping through Efficient Rendering</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06660v1">MapGS: Generalizable Pretraining and Data Augmentation for Online Mapping via Novel View Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      Online mapping reduces the reliance of autonomous vehicles on high-definition (HD) maps, significantly enhancing scalability. However, recent advancements often overlook cross-sensor configuration generalization, leading to performance degradation when models are deployed on vehicles with different camera intrinsics and extrinsics. With the rapid evolution of novel view synthesis methods, we investigate the extent to which these techniques can be leveraged to address the sensor configuration generalization challenge. We propose a novel framework leveraging Gaussian splatting to reconstruct scenes and render camera images in target sensor configurations. The target config sensor data, along with labels mapped to the target config, are used to train online mapping models. Our proposed framework on the nuScenes and Argoverse 2 datasets demonstrates a performance improvement of 18% through effective dataset augmentation, achieves faster convergence and efficient training, and exceeds state-of-the-art performance when using only 25% of the original training data. This enables data reuse and reduces the need for laborious data labeling. Project page at https://henryzhangzhy.github.io/mapgs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06488v1">NVS-SQA: Exploring Self-Supervised Quality Representation Learning for Neurally Synthesized Scenes without References</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      Neural View Synthesis (NVS), such as NeRF and 3D Gaussian Splatting, effectively creates photorealistic scenes from sparse viewpoints, typically evaluated by quality assessment methods like PSNR, SSIM, and LPIPS. However, these full-reference methods, which compare synthesized views to reference views, may not fully capture the perceptual quality of neurally synthesized scenes (NSS), particularly due to the limited availability of dense reference views. Furthermore, the challenges in acquiring human perceptual labels hinder the creation of extensive labeled datasets, risking model overfitting and reduced generalizability. To address these issues, we propose NVS-SQA, a NSS quality assessment method to learn no-reference quality representations through self-supervision without reliance on human labels. Traditional self-supervised learning predominantly relies on the "same instance, similar representation" assumption and extensive datasets. However, given that these conditions do not apply in NSS quality assessment, we employ heuristic cues and quality scores as learning objectives, along with a specialized contrastive pair preparation process to improve the effectiveness and efficiency of learning. The results show that NVS-SQA outperforms 17 no-reference methods by a large margin (i.e., on average 109.5% in SRCC, 98.6% in PLCC, and 91.5% in KRCC over the second best) and even exceeds 16 full-reference methods across all evaluation metrics (i.e., 22.9% in SRCC, 19.1% in PLCC, and 18.6% in KRCC over the second best).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02751v3">Splat-Nav: Safe Real-Time Robot Navigation in Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      We present Splat-Nav, a real-time robot navigation pipeline for Gaussian Splatting (GSplat) scenes, a powerful new 3D scene representation. Splat-Nav consists of two components: 1) Splat-Plan, a safe planning module, and 2) Splat-Loc, a robust vision-based pose estimation module. Splat-Plan builds a safe-by-construction polytope corridor through the map based on mathematically rigorous collision constraints and then constructs a B\'ezier curve trajectory through this corridor. Splat-Loc provides real-time recursive state estimates given only an RGB feed from an on-board camera, leveraging the point-cloud representation inherent in GSplat scenes. Working together, these modules give robots the ability to recursively re-plan smooth and safe trajectories to goal locations. Goals can be specified with position coordinates, or with language commands by using a semantic GSplat. We demonstrate improved safety compared to point cloud-based methods in extensive simulation experiments. In a total of 126 hardware flights, we demonstrate equivalent safety and speed compared to motion capture and visual odometry, but without a manual frame alignment required by those methods. We show online re-planning at more than 2 Hz and pose estimation at about 25 Hz, an order of magnitude faster than Neural Radiance Field (NeRF)-based navigation methods, thereby enabling real-time navigation. We provide experiment videos on our project page at https://chengine.github.io/splatnav/. Our codebase and ROS nodes can be found at https://github.com/chengine/splatnav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.02751v3">Splat-Nav: Safe Real-Time Robot Navigation in Gaussian Splatting Maps</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      We present Splat-Nav, a real-time robot navigation pipeline for Gaussian Splatting (GSplat) scenes, a powerful new 3D scene representation. Splat-Nav consists of two components: 1) Splat-Plan, a safe planning module, and 2) Splat-Loc, a robust vision-based pose estimation module. Splat-Plan builds a safe-by-construction polytope corridor through the map based on mathematically rigorous collision constraints and then constructs a Bézier curve trajectory through this corridor. Splat-Loc provides real-time recursive state estimates given only an RGB feed from an on-board camera, leveraging the point-cloud representation inherent in GSplat scenes. Working together, these modules give robots the ability to recursively re-plan smooth and safe trajectories to goal locations. Goals can be specified with position coordinates, or with language commands by using a semantic GSplat. We demonstrate improved safety compared to point cloud-based methods in extensive simulation experiments. In a total of 126 hardware flights, we demonstrate equivalent safety and speed compared to motion capture and visual odometry, but without a manual frame alignment required by those methods. We show online re-planning at more than 2 Hz and pose estimation at about 25 Hz, an order of magnitude faster than Neural Radiance Field (NeRF)-based navigation methods, thereby enabling real-time navigation. We provide experiment videos on our project page at https://chengine.github.io/splatnav/. Our codebase and ROS nodes can be found at https://github.com/chengine/splatnav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06521v2">PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 project page: https://zju3dv.github.io/pgsr/
    </div>
    <details class="paper-abstract">
      Recently, 3D Gaussian Splatting (3DGS) has attracted widespread attention due to its high-quality rendering, and ultra-fast training and rendering speed. However, due to the unstructured and irregular nature of Gaussian point clouds, it is difficult to guarantee geometric reconstruction accuracy and multi-view consistency simply by relying on image reconstruction loss. Although many studies on surface reconstruction based on 3DGS have emerged recently, the quality of their meshes is generally unsatisfactory. To address this problem, we propose a fast planar-based Gaussian splatting reconstruction representation (PGSR) to achieve high-fidelity surface reconstruction while ensuring high-quality rendering. Specifically, we first introduce an unbiased depth rendering method, which directly renders the distance from the camera origin to the Gaussian plane and the corresponding normal map based on the Gaussian distribution of the point cloud, and divides the two to obtain the unbiased depth. We then introduce single-view geometric, multi-view photometric, and geometric regularization to preserve global geometric accuracy. We also propose a camera exposure compensation model to cope with scenes with large illumination variations. Experiments on indoor and outdoor scenes show that our method achieves fast training and rendering while maintaining high-fidelity rendering and geometric reconstruction, outperforming 3DGS-based and NeRF-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05757v1">Locality-aware Gaussian Compression for Fast and High-quality Rendering</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 28 pages, 15 figures, and 14 tables
    </div>
    <details class="paper-abstract">
      We present LocoGS, a locality-aware 3D Gaussian Splatting (3DGS) framework that exploits the spatial coherence of 3D Gaussians for compact modeling of volumetric scenes. To this end, we first analyze the local coherence of 3D Gaussian attributes, and propose a novel locality-aware 3D Gaussian representation that effectively encodes locally-coherent Gaussian attributes using a neural field representation with a minimal storage requirement. On top of the novel representation, LocoGS is carefully designed with additional components such as dense initialization, an adaptive spherical harmonics bandwidth scheme and different encoding schemes for different Gaussian attributes to maximize compression performance. Experimental results demonstrate that our approach outperforms the rendering quality of existing compact Gaussian representations for representative real-world 3D datasets while achieving from 54.6$\times$ to 96.6$\times$ compressed storage size and from 2.1$\times$ to 2.4$\times$ rendering speed than 3DGS. Even our approach also demonstrates an averaged 2.4$\times$ higher rendering speed than the state-of-the-art compression method with comparable compression performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05427v1">Zero-1-to-G: Taming Pretrained 2D Diffusion Model for Direct 3D Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      Recent advances in 2D image generation have achieved remarkable quality,largely driven by the capacity of diffusion models and the availability of large-scale datasets. However, direct 3D generation is still constrained by the scarcity and lower fidelity of 3D datasets. In this paper, we introduce Zero-1-to-G, a novel approach that addresses this problem by enabling direct single-view generation on Gaussian splats using pretrained 2D diffusion models. Our key insight is that Gaussian splats, a 3D representation, can be decomposed into multi-view images encoding different attributes. This reframes the challenging task of direct 3D generation within a 2D diffusion framework, allowing us to leverage the rich priors of pretrained 2D diffusion models. To incorporate 3D awareness, we introduce cross-view and cross-attribute attention layers, which capture complex correlations and enforce 3D consistency across generated splats. This makes Zero-1-to-G the first direct image-to-3D generative model to effectively utilize pretrained 2D diffusion priors, enabling efficient training and improved generalization to unseen objects. Extensive experiments on both synthetic and in-the-wild datasets demonstrate superior performance in 3D object generation, offering a new approach to high-quality 3D generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05242v1">Scaffold-SLAM: Structured 3D Gaussians for Simultaneous Localization and Photorealistic Mapping</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has recently revolutionized novel view synthesis in the Simultaneous Localization and Mapping (SLAM). However, existing SLAM methods utilizing 3DGS have failed to provide high-quality novel view rendering for monocular, stereo, and RGB-D cameras simultaneously. Notably, some methods perform well for RGB-D cameras but suffer significant degradation in rendering quality for monocular cameras. In this paper, we present Scaffold-SLAM, which delivers simultaneous localization and high-quality photorealistic mapping across monocular, stereo, and RGB-D cameras. We introduce two key innovations to achieve this state-of-the-art visual quality. First, we propose Appearance-from-Motion embedding, enabling 3D Gaussians to better model image appearance variations across different camera poses. Second, we introduce a frequency regularization pyramid to guide the distribution of Gaussians, allowing the model to effectively capture finer details in the scene. Extensive experiments on monocular, stereo, and RGB-D datasets demonstrate that Scaffold-SLAM significantly outperforms state-of-the-art methods in photorealistic mapping quality, e.g., PSNR is 16.76% higher in the TUM RGB-D datasets for monocular cameras.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04782v1">GaussianVideo: Efficient Video Representation via Hierarchical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 10 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Efficient neural representations for dynamic video scenes are critical for applications ranging from video compression to interactive simulations. Yet, existing methods often face challenges related to high memory usage, lengthy training times, and temporal consistency. To address these issues, we introduce a novel neural video representation that combines 3D Gaussian splatting with continuous camera motion modeling. By leveraging Neural ODEs, our approach learns smooth camera trajectories while maintaining an explicit 3D scene representation through Gaussians. Additionally, we introduce a spatiotemporal hierarchical learning strategy, progressively refining spatial and temporal features to enhance reconstruction quality and accelerate convergence. This memory-efficient approach achieves high-quality rendering at impressive speeds. Experimental results show that our hierarchical learning, combined with robust camera motion modeling, captures complex dynamic scenes with strong temporal consistency, achieving state-of-the-art performance across diverse video datasets in both high- and low-motion scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04628v1">FatesGS: Fast and Accurate Sparse-View Surface Reconstruction using Gaussian Splatting with Depth-Feature Consistency</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 Accepted by AAAI 2025. Project page: https://alvin528.github.io/FatesGS/
    </div>
    <details class="paper-abstract">
      Recently, Gaussian Splatting has sparked a new trend in the field of computer vision. Apart from novel view synthesis, it has also been extended to the area of multi-view reconstruction. The latest methods facilitate complete, detailed surface reconstruction while ensuring fast training speed. However, these methods still require dense input views, and their output quality significantly degrades with sparse views. We observed that the Gaussian primitives tend to overfit the few training views, leading to noisy floaters and incomplete reconstruction surfaces. In this paper, we present an innovative sparse-view reconstruction framework that leverages intra-view depth and multi-view feature consistency to achieve remarkably accurate surface reconstruction. Specifically, we utilize monocular depth ranking information to supervise the consistency of depth distribution within patches and employ a smoothness loss to enhance the continuity of the distribution. To achieve finer surface reconstruction, we optimize the absolute position of depth through multi-view projection features. Extensive experiments on DTU and BlendedMVS demonstrate that our method outperforms state-of-the-art methods with a speedup of 60x to 200x, achieving swift and fine-grained mesh reconstruction without the need for costly pre-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17378v3">Balanced 3DGS: Gaussian-wise Parallelism Rendering with Fine-Grained Tiling</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) is increasingly attracting attention in both academia and industry owing to its superior visual quality and rendering speed. However, training a 3DGS model remains a time-intensive task, especially in load imbalance scenarios where workload diversity among pixels and Gaussian spheres causes poor renderCUDA kernel performance. We introduce Balanced 3DGS, a Gaussian-wise parallelism rendering with fine-grained tiling approach in 3DGS training process, perfectly solving load-imbalance issues. First, we innovatively introduce the inter-block dynamic workload distribution technique to map workloads to Streaming Multiprocessor(SM) resources within a single GPU dynamically, which constitutes the foundation of load balancing. Second, we are the first to propose the Gaussian-wise parallel rendering technique to significantly reduce workload divergence inside a warp, which serves as a critical component in addressing load imbalance. Based on the above two methods, we further creatively put forward the fine-grained combined load balancing technique to uniformly distribute workload across all SMs, which boosts the forward renderCUDA kernel performance by up to 7.52x. Besides, we present a self-adaptive render kernel selection strategy during the 3DGS training process based on different load-balance situations, which effectively improves training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00625v2">Gaussian Building Mesh (GBM): Extract a Building's 3D Mesh with Google Earth and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Recently released open-source pre-trained foundational image segmentation and object detection models (SAM2+GroundingDINO) allow for geometrically consistent segmentation of objects of interest in multi-view 2D images. Users can use text-based or click-based prompts to segment objects of interest without requiring labeled training datasets. Gaussian Splatting allows for the learning of the 3D representation of a scene's geometry and radiance based on 2D images. Combining Google Earth Studio, SAM2+GroundingDINO, 2D Gaussian Splatting, and our improvements in mask refinement based on morphological operations and contour simplification, we created a pipeline to extract the 3D mesh of any building based on its name, address, or geographic coordinates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03875v1">ZDySS -- Zero-Shot Dynamic Scene Stylization using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Stylizing a dynamic scene based on an exemplar image is critical for various real-world applications, including gaming, filmmaking, and augmented and virtual reality. However, achieving consistent stylization across both spatial and temporal dimensions remains a significant challenge. Most existing methods are designed for static scenes and often require an optimization process for each style image, limiting their adaptability. We introduce ZDySS, a zero-shot stylization framework for dynamic scenes, allowing our model to generalize to previously unseen style images at inference. Our approach employs Gaussian splatting for scene representation, linking each Gaussian to a learned feature vector that renders a feature map for any given view and timestamp. By applying style transfer on the learned feature vectors instead of the rendered feature map, we enhance spatio-temporal consistency across frames. Our method demonstrates superior performance and coherence over state-of-the-art baselines in tests on real-world dynamic scenes, making it a robust solution for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03714v1">MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 The last two authors are co-corresponding authors. Please visit our project page at https://kaist-viclab.github.io/MoDecGS-site/
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has made significant strides in scene representation and neural rendering, with intense efforts focused on adapting it for dynamic scenes. Despite delivering remarkable rendering quality and speed, existing methods struggle with storage demands and representing complex real-world motions. To tackle these issues, we propose MoDecGS, a memory-efficient Gaussian splatting framework designed for reconstructing novel views in challenging scenarios with complex motions. We introduce GlobaltoLocal Motion Decomposition (GLMD) to effectively capture dynamic motions in a coarsetofine manner. This approach leverages Global Canonical Scaffolds (Global CS) and Local Canonical Scaffolds (Local CS), extending static Scaffold representation to dynamic video reconstruction. For Global CS, we propose Global Anchor Deformation (GAD) to efficiently represent global dynamics along complex motions, by directly deforming the implicit Scaffold attributes which are anchor position, offset, and local context features. Next, we finely adjust local motions via the Local Gaussian Deformation (LGD) of Local CS explicitly. Additionally, we introduce Temporal Interval Adjustment (TIA) to automatically control the temporal coverage of each Local CS during training, allowing MoDecGS to find optimal interval assignments based on the specified number of temporal segments. Extensive evaluations demonstrate that MoDecGS achieves an average 70% reduction in model size over stateoftheart methods for dynamic 3D Gaussians from realworld dynamic videos while maintaining or even improving rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03605v1">ConcealGS: Concealing Invisible Copyright Information in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      With the rapid development of 3D reconstruction technology, the widespread distribution of 3D data has become a future trend. While traditional visual data (such as images and videos) and NeRF-based formats already have mature techniques for copyright protection, steganographic techniques for the emerging 3D Gaussian Splatting (3D-GS) format have yet to be fully explored. To address this, we propose ConcealGS, an innovative method for embedding implicit information into 3D-GS. By introducing the knowledge distillation and gradient optimization strategy based on 3D-GS, ConcealGS overcomes the limitations of NeRF-based models and enhances the robustness of implicit information and the quality of 3D reconstruction. We evaluate ConcealGS in various potential application scenarios, and experimental results have demonstrated that ConcealGS not only successfully recovers implicit information but also has almost no impact on rendering quality, providing a new approach for embedding invisible and recoverable information into 3D models in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.19370v2">BeSplat: Gaussian Splatting from a Single Blurry Image and Event Stream</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 Accepted for publication at EVGEN2025, WACV-25 Workshop
    </div>
    <details class="paper-abstract">
      Novel view synthesis has been greatly enhanced by the development of radiance field methods. The introduction of 3D Gaussian Splatting (3DGS) has effectively addressed key challenges, such as long training times and slow rendering speeds, typically associated with Neural Radiance Fields (NeRF), while maintaining high-quality reconstructions. In this work (BeSplat), we demonstrate the recovery of sharp radiance field (Gaussian splats) from a single motion-blurred image and its corresponding event stream. Our method jointly learns the scene representation via Gaussian Splatting and recovers the camera motion through Bezier SE(3) formulation effectively, minimizing discrepancies between synthesized and real-world measurements of both blurry image and corresponding event stream. We evaluate our approach on both synthetic and real datasets, showcasing its ability to render view-consistent, sharp images from the learned radiance field and the estimated camera trajectory. To the best of our knowledge, ours is the first work to address this highly challenging ill-posed problem in a Gaussian Splatting framework with the effective incorporation of temporal information captured using the event stream.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03399v1">Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is a recognized method for 3D scene representation, known for its high rendering quality and speed. However, its substantial data requirements present challenges for practical applications. In this paper, we introduce an efficient compression technique that significantly reduces storage overhead by using compact representation. We propose a unified architecture that combines point cloud data and feature planes through a progressive tri-plane structure. Our method utilizes 2D feature planes, enabling continuous spatial representation. To further optimize these representations, we incorporate entropy modeling in the frequency domain, specifically designed for standard video codecs. We also propose channel-wise bit allocation to achieve a better trade-off between bitrate consumption and feature plane representation. Consequently, our model effectively leverages spatial correlations within the feature planes to enhance rate-distortion performance using standard, non-differentiable video codecs. Experimental results demonstrate that our method outperforms existing methods in data compactness while maintaining high rendering quality. Our project page is available at https://fraunhoferhhi.github.io/CodecGS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03229v1">Gaussian Masked Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      This paper explores Masked Autoencoders (MAE) with Gaussian Splatting. While reconstructive self-supervised learning frameworks such as MAE learns good semantic abstractions, it is not trained for explicit spatial awareness. Our approach, named Gaussian Masked Autoencoder, or GMAE, aims to learn semantic abstractions and spatial understanding jointly. Like MAE, it reconstructs the image end-to-end in the pixel space, but beyond MAE, it also introduces an intermediate, 3D Gaussian-based representation and renders images via splatting. We show that GMAE can enable various zero-shot learning capabilities of spatial understanding (e.g., figure-ground segmentation, image layering, edge detection, etc.) while preserving the high-level semantics of self-supervised representation quality from MAE. To our knowledge, we are the first to employ Gaussian primitives in an image representation learning framework beyond optimization-based single-scene reconstructions. We believe GMAE will inspire further research in this direction and contribute to developing next-generation techniques for modeling high-fidelity visual data. More details at https://brjathu.github.io/gmae
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02845v1">HOGSA: Bimanual Hand-Object Interaction Understanding with 3D Gaussian Splatting Based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 Accepted by AAAI2025
    </div>
    <details class="paper-abstract">
      Understanding of bimanual hand-object interaction plays an important role in robotics and virtual reality. However, due to significant occlusions between hands and object as well as the high degree-of-freedom motions, it is challenging to collect and annotate a high-quality, large-scale dataset, which prevents further improvement of bimanual hand-object interaction-related baselines. In this work, we propose a new 3D Gaussian Splatting based data augmentation framework for bimanual hand-object interaction, which is capable of augmenting existing dataset to large-scale photorealistic data with various hand-object pose and viewpoints. First, we use mesh-based 3DGS to model objects and hands, and to deal with the rendering blur problem due to multi-resolution input images used, we design a super-resolution module. Second, we extend the single hand grasping pose optimization module for the bimanual hand object to generate various poses of bimanual hand-object interaction, which can significantly expand the pose distribution of the dataset. Third, we conduct an analysis for the impact of different aspects of the proposed data augmentation on the understanding of the bimanual hand-object interaction. We perform our data augmentation on two benchmarks, H2O and Arctic, and verify that our method can improve the performance of the baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10133v2">Efficient Density Control for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) excels in novel view synthesis, balancing advanced rendering quality with real-time performance. However, in trained scenes, a large number of Gaussians with low opacity significantly increase rendering costs. This issue arises due to flaws in the split and clone operations during the densification process, which lead to extensive Gaussian overlap and subsequent opacity reduction. To enhance the efficiency of Gaussian utilization, we improve the adaptive density control of 3DGS. First, we introduce a more efficient long-axis split operation to replace the original clone and split, which mitigates Gaussian overlap and improves densification efficiency.Second, we propose a simple adaptive pruning technique to reduce the number of low-opacity Gaussians. Finally, by dynamically lowering the splitting threshold and applying importance weighting, the efficiency of Gaussian utilization is further improved. We evaluate our proposed method on various challenging real-world datasets. Experimental results show that our Efficient Density Control (EDC) can enhance both the rendering speed and quality. Code is available at https://github.com/XiaoBin2001/EDC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02690v1">GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 Project Page: https://wkbian.github.io/Projects/GS-DiT/
    </div>
    <details class="paper-abstract">
      4D video control is essential in video generation as it enables the use of sophisticated lens techniques, such as multi-camera shooting and dolly zoom, which are currently unsupported by existing methods. Training a video Diffusion Transformer (DiT) directly to control 4D content requires expensive multi-view videos. Inspired by Monocular Dynamic novel View Synthesis (MDVS) that optimizes a 4D representation and renders videos according to different 4D elements, such as camera pose and object motion editing, we bring pseudo 4D Gaussian fields to video generation. Specifically, we propose a novel framework that constructs a pseudo 4D Gaussian field with dense 3D point tracking and renders the Gaussian field for all video frames. Then we finetune a pretrained DiT to generate videos following the guidance of the rendered video, dubbed as GS-DiT. To boost the training of the GS-DiT, we also propose an efficient Dense 3D Point Tracking (D3D-PT) method for the pseudo 4D Gaussian field construction. Our D3D-PT outperforms SpatialTracker, the state-of-the-art sparse 3D point tracking method, in accuracy and accelerates the inference speed by two orders of magnitude. During the inference stage, GS-DiT can generate videos with the same dynamic content while adhering to different camera parameters, addressing a significant limitation of current video generation models. GS-DiT demonstrates strong generalization capabilities and extends the 4D controllability of Gaussian splatting to video generation beyond just camera poses. It supports advanced cinematic effects through the manipulation of the Gaussian field and camera intrinsics, making it a powerful tool for creative video production. Demos are available at https://wkbian.github.io/Projects/GS-DiT/.
    </details>
</div>
