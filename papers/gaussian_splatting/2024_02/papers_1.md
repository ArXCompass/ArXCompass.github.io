# gaussian splatting - 2024_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.19441v1">3D Gaussian Model for Animation and Texturing</a></div>
    <div class="paper-meta">
      📅 2024-02-29
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting has made a marked impact on neural rendering by achieving impressive fidelity and performance. Despite this achievement, however, it is not readily applicable to developing interactive applications. Real-time applications like XR apps and games require functions such as animation, UV-mapping, and model editing simultaneously manipulated through the usage of a 3D model. We propose a modeling that is analogous to typical 3D models, which we call 3D Gaussian Model (3DGM); it provides a manipulatable proxy for novel animation and texture transfer. By binding the 3D Gaussians in texture space and re-projecting them back to world space through implicit shell mapping, we show how our 3D modeling can serve as a valid rendering methodology for interactive applications. It is further noted that recently, 3D mesh reconstruction works have been able to produce high-quality mesh for rendering. Our work, on the other hand, only requires an approximated geometry for rendering an object in high fidelity. Applicationwise, we will show that our proxy-based 3DGM is capable of driving novel animation without animated training data and texture transferring via UV mapping of the 3D Gaussians. We believe the result indicates the potential of our work for enabling interactive applications for 3D Gaussian Splatting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17427v1">VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-02-27
      | 💬 Accepted to CVPR 2024. Project website: https://vastgaussian.github.io
    </div>
    <details class="paper-abstract">
      Existing NeRF-based methods for large scene reconstruction often have limitations in visual quality and rendering speed. While the recent 3D Gaussian Splatting works well on small-scale and object-centric scenes, scaling it up to large scenes poses challenges due to limited video memory, long optimization time, and noticeable appearance variations. To address these challenges, we present VastGaussian, the first method for high-quality reconstruction and real-time rendering on large scenes based on 3D Gaussian Splatting. We propose a progressive partitioning strategy to divide a large scene into multiple cells, where the training cameras and point cloud are properly distributed with an airspace-aware visibility criterion. These cells are merged into a complete scene after parallel optimization. We also introduce decoupled appearance modeling into the optimization process to reduce appearance variations in the rendered images. Our approach outperforms existing NeRF-based methods and achieves state-of-the-art results on multiple large scene datasets, enabling fast optimization and high-fidelity real-time rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14650v1">GaussianPro: 3D Gaussian Splatting with Progressive Propagation</a></div>
    <div class="paper-meta">
      📅 2024-02-22
      | 💬 See the project page for code, data: https://kcheng1021.github.io/gaussianpro.github.io
    </div>
    <details class="paper-abstract">
      The advent of 3D Gaussian Splatting (3DGS) has recently brought about a revolution in the field of neural rendering, facilitating high-quality renderings at real-time speed. However, 3DGS heavily depends on the initialized point cloud produced by Structure-from-Motion (SfM) techniques. When tackling with large-scale scenes that unavoidably contain texture-less surfaces, the SfM techniques always fail to produce enough points in these surfaces and cannot provide good initialization for 3DGS. As a result, 3DGS suffers from difficult optimization and low-quality renderings. In this paper, inspired by classical multi-view stereo (MVS) techniques, we propose GaussianPro, a novel method that applies a progressive propagation strategy to guide the densification of the 3D Gaussians. Compared to the simple split and clone strategies used in 3DGS, our method leverages the priors of the existing reconstructed geometries of the scene and patch matching techniques to produce new Gaussians with accurate positions and orientations. Experiments on both large-scale and small-scale scenes validate the effectiveness of our method, where our method significantly surpasses 3DGS on the Waymo dataset, exhibiting an improvement of 1.15dB in terms of PSNR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10642v3">Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2024-02-22
      | 💬 ICLR 2024
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic 3D scenes from 2D images and generating diverse views over time is challenging due to scene complexity and temporal dynamics. Despite advancements in neural implicit models, limitations persist: (i) Inadequate Scene Structure: Existing methods struggle to reveal the spatial and temporal structure of dynamic scenes from directly learning the complex 6D plenoptic function. (ii) Scaling Deformation Modeling: Explicitly modeling scene element deformation becomes impractical for complex dynamics. To address these issues, we consider the spacetime as an entirety and propose to approximate the underlying spatio-temporal 4D volume of a dynamic scene by optimizing a collection of 4D primitives, with explicit geometry and appearance modeling. Learning to optimize the 4D primitives enables us to synthesize novel views at any desired time with our tailored rendering routine. Our model is conceptually simple, consisting of a 4D Gaussian parameterized by anisotropic ellipses that can rotate arbitrarily in space and time, as well as view-dependent and time-evolved appearance represented by the coefficient of 4D spherindrical harmonics. This approach offers simplicity, flexibility for variable-length video and end-to-end training, and efficient real-time rendering, making it suitable for capturing complex dynamic scene motions. Experiments across various benchmarks, including monocular and multi-view scenarios, demonstrate our 4DGS model's superior visual quality and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.13681v2">Compact 3D Gaussian Representation for Radiance Field</a></div>
    <div class="paper-meta">
      📅 2024-02-15
      | 💬 Project page: http://maincold2.github.io/c3dgs/
    </div>
    <details class="paper-abstract">
      Neural Radiance Fields (NeRFs) have demonstrated remarkable potential in capturing complex 3D scenes with high fidelity. However, one persistent challenge that hinders the widespread adoption of NeRFs is the computational bottleneck due to the volumetric rendering. On the other hand, 3D Gaussian splatting (3DGS) has recently emerged as an alternative representation that leverages a 3D Gaussisan-based representation and adopts the rasterization pipeline to render the images rather than volumetric rendering, achieving very fast rendering speed and promising image quality. However, a significant drawback arises as 3DGS entails a substantial number of 3D Gaussians to maintain the high fidelity of the rendered images, which requires a large amount of memory and storage. To address this critical issue, we place a specific emphasis on two key objectives: reducing the number of Gaussian points without sacrificing performance and compressing the Gaussian attributes, such as view-dependent color and covariance. To this end, we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance. In addition, we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics. Finally, we learn codebooks to compactly represent the geometric attributes of Gaussian by vector quantization. With model compression techniques such as quantization and entropy coding, we consistently show over 25$\times$ reduced storage and enhanced rendering speed, while maintaining the quality of the scene representation, compared to 3DGS. Our work provides a comprehensive framework for 3D scene representation, achieving high performance, fast training, compactness, and real-time rendering. Our project page is available at https://maincold2.github.io/c3dgs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08682v1">IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation</a></div>
    <div class="paper-meta">
      📅 2024-02-13
    </div>
    <details class="paper-abstract">
      Most text-to-3D generators build upon off-the-shelf text-to-image models trained on billions of images. They use variants of Score Distillation Sampling (SDS), which is slow, somewhat unstable, and prone to artifacts. A mitigation is to fine-tune the 2D generator to be multi-view aware, which can help distillation or can be combined with reconstruction networks to output 3D objects directly. In this paper, we further explore the design space of text-to-3D models. We significantly improve multi-view generation by considering video instead of image generators. Combined with a 3D reconstruction algorithm which, by using Gaussian splatting, can optimize a robust image-based loss, we directly produce high-quality 3D outputs from the generated views. Our new method, IM-3D, reduces the number of evaluations of the 2D generator network 10-100x, resulting in a much more efficient pipeline, better quality, fewer geometric inconsistencies, and higher yield of usable 3D assets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06198v2">GS-CLIP: Gaussian Splatting for Contrastive Language-Image-3D Pretraining from Real-World Data</a></div>
    <div class="paper-meta">
      📅 2024-02-13
      | 💬 The content of the technical report needs to be updated and retracted to avoid other impacts
    </div>
    <details class="paper-abstract">
      3D Shape represented as point cloud has achieve advancements in multimodal pre-training to align image and language descriptions, which is curial to object identification, classification, and retrieval. However, the discrete representations of point cloud lost the object's surface shape information and creates a gap between rendering results and 2D correspondences. To address this problem, we propose GS-CLIP for the first attempt to introduce 3DGS (3D Gaussian Splatting) into multimodal pre-training to enhance 3D representation. GS-CLIP leverages a pre-trained vision-language model for a learned common visual and textual space on massive real world image-text pairs and then learns a 3D Encoder for aligning 3DGS optimized per object. Additionally, a novel Gaussian-Aware Fusion is proposed to extract and fuse global explicit feature. As a general framework for language-image-3D pre-training, GS-CLIP is agnostic to 3D backbone networks. Experiments on challenging shows that GS-CLIP significantly improves the state-of-the-art, outperforming the previously best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.12561v2">EndoGaussian: Real-time Gaussian Splatting for Dynamic Endoscopic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2024-02-13
    </div>
    <details class="paper-abstract">
      Reconstructing deformable tissues from endoscopic videos is essential in many downstream surgical applications. However, existing methods suffer from slow rendering speed, greatly limiting their practical use. In this paper, we introduce EndoGaussian, a real-time endoscopic scene reconstruction framework built on 3D Gaussian Splatting (3DGS). By integrating the efficient Gaussian representation and highly-optimized rendering engine, our framework significantly boosts the rendering speed to a real-time level. To adapt 3DGS for endoscopic scenes, we propose two strategies, Holistic Gaussian Initialization (HGI) and Spatio-temporal Gaussian Tracking (SGT), to handle the non-trivial Gaussian initialization and tissue deformation problems, respectively. In HGI, we leverage recent depth estimation models to predict depth maps of input binocular/monocular image sequences, based on which pixels are re-projected and combined for holistic initialization. In SPT, we propose to model surface dynamics using a deformation field, which is composed of an efficient encoding voxel and a lightweight deformation decoder, allowing for Gaussian tracking with minor training and rendering burden. Experiments on public datasets demonstrate our efficacy against prior SOTAs in many aspects, including better rendering speed (195 FPS real-time, 100$\times$ gain), better rendering quality (37.848 PSNR), and less training overhead (within 2 min/scene), showing significant promise for intraoperative surgery applications. Code is available at: \url{https://yifliu3.github.io/EndoGaussian/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04796v1">Mesh-based Gaussian Splatting for Real-time Large-scale Deformation</a></div>
    <div class="paper-meta">
      📅 2024-02-07
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Neural implicit representations, including Neural Distance Fields and Neural Radiance Fields, have demonstrated significant capabilities for reconstructing surfaces with complicated geometry and topology, and generating novel views of a scene. Nevertheless, it is challenging for users to directly deform or manipulate these implicit representations with large deformations in the real-time fashion. Gaussian Splatting(GS) has recently become a promising method with explicit geometry for representing static scenes and facilitating high-quality and real-time synthesis of novel views. However,it cannot be easily deformed due to the use of discrete Gaussians and lack of explicit topology. To address this, we develop a novel GS-based method that enables interactive deformation. Our key idea is to design an innovative mesh-based GS representation, which is integrated into Gaussian learning and manipulation. 3D Gaussians are defined over an explicit mesh, and they are bound with each other: the rendering of 3D Gaussians guides the mesh face split for adaptive refinement, and the mesh face split directs the splitting of 3D Gaussians. Moreover, the explicit mesh constraints help regularize the Gaussian distribution, suppressing poor-quality Gaussians(e.g. misaligned Gaussians,long-narrow shaped Gaussians), thus enhancing visual quality and avoiding artifacts during deformation. Based on this representation, we further introduce a large-scale Gaussian deformation technique to enable deformable GS, which alters the parameters of 3D Gaussians according to the manipulation of the associated mesh. Our method benefits from existing mesh deformation datasets for more realistic data-driven Gaussian deformation. Extensive experiments show that our approach achieves high-quality reconstruction and effective deformation, while maintaining the promising rendering results at a high frame rate(65 FPS on average).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03723v1">Rig3DGS: Creating Controllable Portraits from Casual Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2024-02-06
    </div>
    <details class="paper-abstract">
      Creating controllable 3D human portraits from casual smartphone videos is highly desirable due to their immense value in AR/VR applications. The recent development of 3D Gaussian Splatting (3DGS) has shown improvements in rendering quality and training efficiency. However, it still remains a challenge to accurately model and disentangle head movements and facial expressions from a single-view capture to achieve high-quality renderings. In this paper, we introduce Rig3DGS to address this challenge. We represent the entire scene, including the dynamic subject, using a set of 3D Gaussians in a canonical space. Using a set of control signals, such as head pose and expressions, we transform them to the 3D space with learned deformations to generate the desired rendering. Our key innovation is a carefully designed deformation method which is guided by a learnable prior derived from a 3D morphable model. This approach is highly efficient in training and effective in controlling facial expressions, head positions, and view synthesis across various captures. We demonstrate the effectiveness of our learned deformation through extensive quantitative and qualitative experiments. The project page can be found at http://shahrukhathar.github.io/2024/02/05/Rig3DGS.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.00763v1">360-GS: Layout-guided Panoramic Gaussian Splatting For Indoor Roaming</a></div>
    <div class="paper-meta">
      📅 2024-02-02
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has recently attracted great attention with real-time and photo-realistic renderings. This technique typically takes perspective images as input and optimizes a set of 3D elliptical Gaussians by splatting them onto the image planes, resulting in 2D Gaussians. However, applying 3D-GS to panoramic inputs presents challenges in effectively modeling the projection onto the spherical surface of ${360^\circ}$ images using 2D Gaussians. In practical applications, input panoramas are often sparse, leading to unreliable initialization of 3D Gaussians and subsequent degradation of 3D-GS quality. In addition, due to the under-constrained geometry of texture-less planes (e.g., walls and floors), 3D-GS struggles to model these flat regions with elliptical Gaussians, resulting in significant floaters in novel views. To address these issues, we propose 360-GS, a novel $360^{\circ}$ Gaussian splatting for a limited set of panoramic inputs. Instead of splatting 3D Gaussians directly onto the spherical surface, 360-GS projects them onto the tangent plane of the unit sphere and then maps them to the spherical projections. This adaptation enables the representation of the projection using Gaussians. We guide the optimization of 360-GS by exploiting layout priors within panoramas, which are simple to obtain and contain strong structural information about the indoor scene. Our experimental results demonstrate that 360-GS allows panoramic rendering and outperforms state-of-the-art methods with fewer artifacts in novel view synthesis, thus providing immersive roaming in indoor scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00763v1">360-GS: Layout-guided Panoramic Gaussian Splatting For Indoor Roaming</a></div>
    <div class="paper-meta">
      📅 2024-02-01
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3D-GS) has recently attracted great attention with real-time and photo-realistic renderings. This technique typically takes perspective images as input and optimizes a set of 3D elliptical Gaussians by splatting them onto the image planes, resulting in 2D Gaussians. However, applying 3D-GS to panoramic inputs presents challenges in effectively modeling the projection onto the spherical surface of ${360^\circ}$ images using 2D Gaussians. In practical applications, input panoramas are often sparse, leading to unreliable initialization of 3D Gaussians and subsequent degradation of 3D-GS quality. In addition, due to the under-constrained geometry of texture-less planes (e.g., walls and floors), 3D-GS struggles to model these flat regions with elliptical Gaussians, resulting in significant floaters in novel views. To address these issues, we propose 360-GS, a novel $360^{\circ}$ Gaussian splatting for a limited set of panoramic inputs. Instead of splatting 3D Gaussians directly onto the spherical surface, 360-GS projects them onto the tangent plane of the unit sphere and then maps them to the spherical projections. This adaptation enables the representation of the projection using Gaussians. We guide the optimization of 360-GS by exploiting layout priors within panoramas, which are simple to obtain and contain strong structural information about the indoor scene. Our experimental results demonstrate that 360-GS allows panoramic rendering and outperforms state-of-the-art methods with fewer artifacts in novel view synthesis, thus providing immersive roaming in indoor scenarios.
    </details>
</div>
