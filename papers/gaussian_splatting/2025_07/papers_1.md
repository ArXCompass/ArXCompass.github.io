# gaussian splatting - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [gaussian splatting](https://arxcompass.github.io/papers/gaussian_splatting)

- Part 1
- [Part 2](papers_2.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21401v3">Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted by ICCV 2025, Code: https://github.com/zhirui-gao/Curve-Gaussian
    </div>
    <details class="paper-abstract">
      This paper presents an end-to-end framework for reconstructing 3D parametric curves directly from multi-view edge maps. Contrasting with existing two-stage methods that follow a sequential ``edge point cloud reconstruction and parametric curve fitting'' pipeline, our one-stage approach optimizes 3D parametric curves directly from 2D edge maps, eliminating error accumulation caused by the inherent optimization gap between disconnected stages. However, parametric curves inherently lack suitability for rendering-based multi-view optimization, necessitating a complementary representation that preserves their geometric properties while enabling differentiable rendering. We propose a novel bi-directional coupling mechanism between parametric curves and edge-oriented Gaussian components. This tight correspondence formulates a curve-aware Gaussian representation, \textbf{CurveGaussian}, that enables differentiable rendering of 3D curves, allowing direct optimization guided by multi-view evidence. Furthermore, we introduce a dynamically adaptive topology optimization framework during training to refine curve structures through linearization, merging, splitting, and pruning operations. Comprehensive evaluations on the ABC dataset and real-world benchmarks demonstrate our one-stage method's superiority over two-stage alternatives, particularly in producing cleaner and more robust reconstructions. Additionally, by directly optimizing parametric curves, our method significantly reduces the parameter count during training, achieving both higher efficiency and superior performance compared to existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15300v2">GCC: A 3DGS Inference Architecture with Gaussian-Wise and Cross-Stage Conditional Processing</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading neural rendering technique for high-fidelity view synthesis, prompting the development of dedicated 3DGS accelerators for mobile applications. Through in-depth analysis, we identify two major limitations in the conventional decoupled preprocessing-rendering dataflow adopted by existing accelerators: 1) a significant portion of preprocessed Gaussians are not used in rendering, and 2) the same Gaussian gets repeatedly loaded across different tile renderings, resulting in substantial computational and data movement overhead. To address these issues, we propose GCC, a novel accelerator designed for fast and energy-efficient 3DGS inference. At the dataflow level, GCC introduces: 1) cross-stage conditional processing, which interleaves preprocessing and rendering to dynamically skip unnecessary Gaussian preprocessing; and 2) Gaussian-wise rendering, ensuring that all rendering operations for a given Gaussian are completed before moving to the next, thereby eliminating duplicated Gaussian loading. We also propose an alpha-based boundary identification method to derive compact and accurate Gaussian regions, thereby reducing rendering costs. We implement our GCC accelerator in 28nm technology. Extensive experiments demonstrate that GCC significantly outperforms the state-of-the-art 3DGS inference accelerator, GSCore, in both performance and energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16535v1">EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Despite the remarkable developments achieved by recent 3D generation works, scaling these methods to geographic extents, such as modeling thousands of square kilometers of Earth's surface, remains an open challenge. We address this through a dual innovation in data infrastructure and model architecture. First, we introduce Aerial-Earth3D, the largest 3D aerial dataset to date, consisting of 50k curated scenes (each measuring 600m x 600m) captured across the U.S. mainland, comprising 45M multi-view Google Earth frames. Each scene provides pose-annotated multi-view images, depth maps, normals, semantic segmentation, and camera poses, with explicit quality control to ensure terrain diversity. Building on this foundation, we propose EarthCrafter, a tailored framework for large-scale 3D Earth generation via sparse-decoupled latent diffusion. Our architecture separates structural and textural generation: 1) Dual sparse 3D-VAEs compress high-resolution geometric voxels and textural 2D Gaussian Splats (2DGS) into compact latent spaces, largely alleviating the costly computation suffering from vast geographic scales while preserving critical information. 2) We propose condition-aware flow matching models trained on mixed inputs (semantics, images, or neither) to flexibly model latent geometry and texture features independently. Extensive experiments demonstrate that EarthCrafter performs substantially better in extremely large-scale generation. The framework further supports versatile applications, from semantic-guided urban layout generation to unconditional terrain synthesis, while maintaining geographic plausibility through our rich data priors from Aerial-Earth3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05182v2">MGSR: 2D/3D Mutual-boosted Gaussian Splatting for High-fidelity Surface Reconstruction under Various Light Conditions</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at ICCV'25
    </div>
    <details class="paper-abstract">
      Novel view synthesis (NVS) and surface reconstruction (SR) are essential tasks in 3D Gaussian Splatting (3D-GS). Despite recent progress, these tasks are often addressed independently, with GS-based rendering methods struggling under diverse light conditions and failing to produce accurate surfaces, while GS-based reconstruction methods frequently compromise rendering quality. This raises a central question: must rendering and reconstruction always involve a trade-off? To address this, we propose MGSR, a 2D/3D Mutual-boosted Gaussian splatting for Surface Reconstruction that enhances both rendering quality and 3D reconstruction accuracy. MGSR introduces two branches--one based on 2D-GS and the other on 3D-GS. The 2D-GS branch excels in surface reconstruction, providing precise geometry information to the 3D-GS branch. Leveraging this geometry, the 3D-GS branch employs a geometry-guided illumination decomposition module that captures reflected and transmitted components, enabling realistic rendering under varied light conditions. Using the transmitted component as supervision, the 2D-GS branch also achieves high-fidelity surface reconstruction. Throughout the optimization process, the 2D-GS and 3D-GS branches undergo alternating optimization, providing mutual supervision. Prior to this, each branch completes an independent warm-up phase, with an early stopping strategy implemented to reduce computational costs. We evaluate MGSR on a diverse set of synthetic and real-world datasets, at both object and scene levels, demonstrating strong performance in rendering and surface reconstruction. Code is available at https://github.com/TsingyuanChou/MGSR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13613v3">MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) has recently emerged as a promising technique for capturing complex dynamic 3D scenes with high fidelity. It utilizes a 4D Gaussian representation and a GPU-friendly rasterizer, enabling rapid rendering speeds. Despite its advantages, 4DGS faces significant challenges, notably the requirement of millions of 4D Gaussians, each with extensive associated attributes, leading to substantial memory and storage cost. This paper introduces a memory-efficient framework for 4DGS. We streamline the color attribute by decomposing it into a per-Gaussian direct color component with only 3 parameters and a shared lightweight alternating current color predictor. This approach eliminates the need for spherical harmonics coefficients, which typically involve up to 144 parameters in classic 4DGS, thereby creating a memory-efficient 4D Gaussian representation. Furthermore, we introduce an entropy-constrained Gaussian deformation technique that uses a deformation field to expand the action range of each Gaussian and integrates an opacity-based entropy loss to limit the number of Gaussians, thus forcing our model to use as few Gaussians as possible to fit a dynamic scene well. With simple half-precision storage and zip compression, our framework achieves a storage reduction by approximately 190$\times$ and 125$\times$ on the Technicolor and Neural 3D Video datasets, respectively, compared to the original 4DGS. Meanwhile, it maintains comparable rendering speeds and scene representation quality, setting a new standard in the field. Code is available at https://github.com/Xinjie-Q/MEGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16406v1">Sparse-View 3D Reconstruction: Recent Advances and Open Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 30 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Sparse-view 3D reconstruction is essential for applications in which dense image acquisition is impractical, such as robotics, augmented/virtual reality (AR/VR), and autonomous systems. In these settings, minimal image overlap prevents reliable correspondence matching, causing traditional methods, such as structure-from-motion (SfM) and multiview stereo (MVS), to fail. This survey reviews the latest advances in neural implicit models (e.g., NeRF and its regularized versions), explicit point-cloud-based approaches (e.g., 3D Gaussian Splatting), and hybrid frameworks that leverage priors from diffusion and vision foundation models (VFMs).We analyze how geometric regularization, explicit shape modeling, and generative inference are used to mitigate artifacts such as floaters and pose ambiguities in sparse-view settings. Comparative results on standard benchmarks reveal key trade-offs between the reconstruction accuracy, efficiency, and generalization. Unlike previous reviews, our survey provides a unified perspective on geometry-based, neural implicit, and generative (diffusion-based) methods. We highlight the persistent challenges in domain generalization and pose-free reconstruction and outline future directions for developing 3D-native generative priors and achieving real-time, unconstrained sparse-view reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16144v1">LongSplat: Online Generalizable 3D Gaussian Splatting from Long Sequence Images</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting achieves high-fidelity novel view synthesis, but its application to online long-sequence scenarios is still limited. Existing methods either rely on slow per-scene optimization or fail to provide efficient incremental updates, hindering continuous performance. In this paper, we propose LongSplat, an online real-time 3D Gaussian reconstruction framework designed for long-sequence image input. The core idea is a streaming update mechanism that incrementally integrates current-view observations while selectively compressing redundant historical Gaussians. Crucial to this mechanism is our Gaussian-Image Representation (GIR), a representation that encodes 3D Gaussian parameters into a structured, image-like 2D format. GIR simultaneously enables efficient fusion of current-view and historical Gaussians and identity-aware redundancy compression. These functions enable online reconstruction and adapt the model to long sequences without overwhelming memory or computational costs. Furthermore, we leverage an existing image compression method to guide the generation of more compact and higher-quality 3D Gaussians. Extensive evaluations demonstrate that LongSplat achieves state-of-the-art efficiency-quality trade-offs in real-time novel view synthesis, delivering real-time reconstruction while reducing Gaussian counts by 44\% compared to existing per-pixel Gaussian prediction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15748v1">Appearance Harmonization via Bilateral Grid Prediction with Transformers for 3DGS</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 10 pages, 3 figures, NeurIPS 2025 under review
    </div>
    <details class="paper-abstract">
      Modern camera pipelines apply extensive on-device processing, such as exposure adjustment, white balance, and color correction, which, while beneficial individually, often introduce photometric inconsistencies across views. These appearance variations violate multi-view consistency and degrade the quality of novel view synthesis. Joint optimization of scene representations and per-image appearance embeddings has been proposed to address this issue, but at the cost of increased computational complexity and slower training. In this work, we propose a transformer-based method that predicts spatially adaptive bilateral grids to correct photometric variations in a multi-view consistent manner, enabling robust cross-scene generalization without the need for scene-specific retraining. By incorporating the learned grids into the 3D Gaussian Splatting pipeline, we improve reconstruction quality while maintaining high training efficiency. Extensive experiments show that our approach outperforms or matches existing scene-specific optimization methods in reconstruction fidelity and convergence speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15690v1">DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15683v1">Hi^2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization for Remote Sensing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Visual relocalization, which estimates the 6-degree-of-freedom (6-DoF) camera pose from query images, is fundamental to remote sensing and UAV applications. Existing methods face inherent trade-offs: image-based retrieval and pose regression approaches lack precision, while structure-based methods that register queries to Structure-from-Motion (SfM) models suffer from computational complexity and limited scalability. These challenges are particularly pronounced in remote sensing scenarios due to large-scale scenes, high altitude variations, and domain gaps of existing visual priors. To overcome these limitations, we leverage 3D Gaussian Splatting (3DGS) as a novel scene representation that compactly encodes both 3D geometry and appearance. We introduce $\mathrm{Hi}^2$-GSLoc, a dual-hierarchical relocalization framework that follows a sparse-to-dense and coarse-to-fine paradigm, fully exploiting the rich semantic information and geometric constraints inherent in Gaussian primitives. To handle large-scale remote sensing scenarios, we incorporate partitioned Gaussian training, GPU-accelerated parallel matching, and dynamic memory management strategies. Our approach consists of two stages: (1) a sparse stage featuring a Gaussian-specific consistent render-aware sampling strategy and landmark-guided detector for robust and accurate initial pose estimation, and (2) a dense stage that iteratively refines poses through coarse-to-fine dense rasterization matching while incorporating reliability verification. Through comprehensive evaluation on simulation data, public datasets, and real flight experiments, we demonstrate that our method delivers competitive localization accuracy, recall rate, and computational efficiency while effectively filtering unreliable pose estimates. The results confirm the effectiveness of our approach for practical remote sensing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12137v2">AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Modeling and rendering dynamic urban driving scenes is crucial for self-driving simulation. Current high-quality methods typically rely on costly manual object tracklet annotations, while self-supervised approaches fail to capture dynamic object motions accurately and decompose scenes properly, resulting in rendering artifacts. We introduce AD-GS, a novel self-supervised framework for high-quality free-viewpoint rendering of driving scenes from a single log. At its core is a novel learnable motion model that integrates locality-aware B-spline curves with global-aware trigonometric functions, enabling flexible yet precise dynamic object modeling. Rather than requiring comprehensive semantic labeling, AD-GS automatically segments scenes into objects and background with the simplified pseudo 2D segmentation, representing objects using dynamic Gaussians and bidirectional temporal visibility masks. Further, our model incorporates visibility reasoning and physically rigid regularization to enhance robustness. Extensive evaluations demonstrate that our annotation-free model significantly outperforms current state-of-the-art annotation-free methods and is competitive with annotation-dependent approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15629v1">Gaussian Splatting with Discretized SDF for Relightable Assets</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) has shown its detailed expressive ability and highly efficient rendering speed in the novel view synthesis (NVS) task. The application to inverse rendering still faces several challenges, as the discrete nature of Gaussian primitives makes it difficult to apply geometry constraints. Recent works introduce the signed distance field (SDF) as an extra continuous representation to regularize the geometry defined by Gaussian primitives. It improves the decomposition quality, at the cost of increasing memory usage and complicating training. Unlike these works, we introduce a discretized SDF to represent the continuous SDF in a discrete manner by encoding it within each Gaussian using a sampled value. This approach allows us to link the SDF with the Gaussian opacity through an SDF-to-opacity transformation, enabling rendering the SDF via splatting and avoiding the computational cost of ray marching.The key challenge is to regularize the discrete samples to be consistent with the underlying SDF, as the discrete representation can hardly apply the gradient-based constraints (\eg Eikonal loss). For this, we project Gaussians onto the zero-level set of SDF and enforce alignment with the surface from splatting, namely a projection-based consistency loss. Thanks to the discretized SDF, our method achieves higher relighting quality, while requiring no extra memory beyond GS and avoiding complex manually designed optimization. The experiments reveal that our method outperforms existing Gaussian-based inverse rendering methods. Our code is available at https://github.com/NK-CS-ZZL/DiscretizedSDF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15602v1">SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Surface reconstruction and novel view rendering from sparse-view images are challenging. Signed Distance Function (SDF)-based methods struggle with fine details, while 3D Gaussian Splatting (3DGS)-based approaches lack global geometry coherence. We propose a novel hybrid method that combines the strengths of both approaches: SDF captures coarse geometry to enhance 3DGS-based rendering, while newly rendered images from 3DGS refine the details of SDF for accurate surface reconstruction. As a result, our method surpasses state-of-the-art approaches in surface reconstruction and novel view synthesis on the DTU and MobileBrick datasets. Code will be released at https://github.com/Gaozihui/SurfaceSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06060v2">VisualSpeaker: Visually-Guided 3D Avatar Lip Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted in International Conference on Computer Vision (ICCV) Workshops
    </div>
    <details class="paper-abstract">
      Realistic, high-fidelity 3D facial animations are crucial for expressive avatar systems in human-computer interaction and accessibility. Although prior methods show promising quality, their reliance on the mesh domain limits their ability to fully leverage the rapid visual innovations seen in 2D computer vision and graphics. We propose VisualSpeaker, a novel method that bridges this gap using photorealistic differentiable rendering, supervised by visual speech recognition, for improved 3D facial animation. Our contribution is a perceptual lip-reading loss, derived by passing photorealistic 3D Gaussian Splatting avatar renders through a pre-trained Visual Automatic Speech Recognition model during training. Evaluation on the MEAD dataset demonstrates that VisualSpeaker improves both the standard Lip Vertex Error metric by 56.1% and the perceptual quality of the generated animations, while retaining the controllability of mesh-driven animation. This perceptual focus naturally supports accurate mouthings, essential cues that disambiguate similar manual signs in sign language avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11061v2">Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Recent advances in 3D neural representations and instance-level editing models have enabled the efficient creation of high-quality 3D content. However, achieving precise local 3D edits remains challenging, especially for Gaussian Splatting, due to inconsistent multi-view 2D part segmentations and inherently ambiguous nature of Score Distillation Sampling (SDS) loss. To address these limitations, we propose RoMaP, a novel local 3D Gaussian editing framework that enables precise and drastic part-level modifications. First, we introduce a robust 3D mask generation module with our 3D-Geometry Aware Label Prediction (3D-GALP), which uses spherical harmonics (SH) coefficients to model view-dependent label variations and soft-label property, yielding accurate and consistent part segmentations across viewpoints. Second, we propose a regularized SDS loss that combines the standard SDS loss with additional regularizers. In particular, an L1 anchor loss is introduced via our Scheduled Latent Mixing and Part (SLaMP) editing method, which generates high-quality part-edited 2D images and confines modifications only to the target region while preserving contextual coherence. Additional regularizers, such as Gaussian prior removal, further improve flexibility by allowing changes beyond the existing context, and robust 3D masking prevents unintended edits. Experimental results demonstrate that our RoMaP achieves state-of-the-art local 3D editing on both reconstructed and generated Gaussian scenes and objects qualitatively and quantitatively, making it possible for more robust and flexible part-level 3D Gaussian editing. Code is available at https://janeyeon.github.io/romap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15454v1">ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting is renowned for its high-fidelity reconstructions and real-time novel view synthesis, yet its lack of semantic understanding limits object-level perception. In this work, we propose ObjectGS, an object-aware framework that unifies 3D scene reconstruction with semantic understanding. Instead of treating the scene as a unified whole, ObjectGS models individual objects as local anchors that generate neural Gaussians and share object IDs, enabling precise object-level reconstruction. During training, we dynamically grow or prune these anchors and optimize their features, while a one-hot ID encoding with a classification loss enforces clear semantic constraints. We show through extensive experiments that ObjectGS not only outperforms state-of-the-art methods on open-vocabulary and panoptic segmentation tasks, but also integrates seamlessly with applications like mesh extraction and scene editing. Project page: https://ruijiezhu94.github.io/ObjectGS_page
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08366v2">In-2-4D: Inbetweening from Two Single-View Images to 4D Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      We propose a new problem, In-2-4D, for generative 4D (i.e., 3D + motion) inbetweening from a minimalistic input setting: two single-view images capturing an object in two distinct motion states. Given two images representing the start and end states of an object in motion, our goal is to generate and reconstruct the motion in 4D. We utilize a video interpolation model to predict the motion, but large frame-to-frame motions can lead to ambiguous interpretations. To overcome this, we employ a hierarchical approach to identify keyframes that are visually close to the input states and show significant motion, then generate smooth fragments between them. For each fragment, we construct the 3D representation of the keyframe using Gaussian Splatting. The temporal frames within the fragment guide the motion, enabling their transformation into dynamic Gaussians through a deformation field. To improve temporal consistency and refine 3D motion, we expand the self-attention of multi-view diffusion across timesteps and apply rigid transformation regularization. Finally, we merge the independently generated 3D motion segments by interpolating boundary deformation fields and optimizing them to align with the guiding video, ensuring smooth and flicker-free transitions. Through extensive qualitative and quantitiave experiments as well as a user study, we show the effectiveness of our method and its components. The project page is available at https://in-2-4d.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15300v1">GCC: A 3DGS Inference Architecture with Gaussian-Wise and Cross-Stage Conditional Processing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged as a leading neural rendering technique for high-fidelity view synthesis, prompting the development of dedicated 3DGS accelerators for mobile applications. Through in-depth analysis, we identify two major limitations in the conventional decoupled preprocessing-rendering dataflow adopted by existing accelerators: 1) a significant portion of preprocessed Gaussians are not used in rendering, and 2) the same Gaussian gets repeatedly loaded across different tile renderings, resulting in substantial computational and data movement overhead. To address these issues, we propose GCC, a novel accelerator designed for fast and energy-efficient 3DGS inference. At the dataflow level, GCC introduces: 1) cross-stage conditional processing, which interleaves preprocessing and rendering to dynamically skip unnecessary Gaussian preprocessing; and 2) Gaussian-wise rendering, ensuring that all rendering operations for a given Gaussian are completed before moving to the next, thereby eliminating duplicated Gaussian loading. We also propose an alpha-based boundary identification method to derive compact and accurate Gaussian regions, thereby reducing rendering costs. We implement our GCC accelerator in 28nm technology. Extensive experiments demonstrate that GCC significantly outperforms the state-of-the-art 3DGS inference accelerator, GSCore, in both performance and energy efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13891v2">PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      COLMAP-free 3D Gaussian Splatting (3D-GS) has recently attracted increasing attention due to its remarkable performance in reconstructing high-quality 3D scenes from unposed images or videos. However, it often struggles to handle scenes with complex camera trajectories as featured by drastic rotation and translation across adjacent camera views, leading to degraded estimation of camera poses and further local minima in joint optimization of camera poses and 3D-GS. We propose PCR-GS, an innovative COLMAP-free 3DGS technique that achieves superior 3D scene modeling and camera pose estimation via camera pose co-regularization. PCR-GS achieves regularization from two perspectives. The first is feature reprojection regularization which extracts view-robust DINO features from adjacent camera views and aligns their semantic information for camera pose regularization. The second is wavelet-based frequency regularization which exploits discrepancy in high-frequency details to further optimize the rotation matrix in camera poses. Extensive experiments over multiple real-world scenes show that the proposed PCR-GS achieves superior pose-free 3D-GS scene modeling under dramatic changes of camera trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12498v2">Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized 3D scene reconstruction, which effectively balances rendering quality, efficiency, and speed. However, existing 3DGS approaches usually generate plausible outputs and face significant challenges in complex scene reconstruction, manifesting as incomplete holistic structural outlines and unclear local lighting effects. To address these issues simultaneously, we propose a novel decoupled optimization framework, which integrates wavelet decomposition into 3D Gaussian Splatting and 2D sampling. Technically, through 3D wavelet decomposition, our approach divides point clouds into high-frequency and low-frequency components, enabling targeted optimization for each. The low-frequency component captures global structural outlines and manages the distribution of Gaussians through voxelization. In contrast, the high-frequency component restores intricate geometric and textural details while incorporating a relight module to mitigate lighting artifacts and enhance photorealistic rendering. Additionally, a 2D wavelet decomposition is applied to the training images, simulating radiance variations. This provides critical guidance for high-frequency detail reconstruction, ensuring seamless integration of details with the global structure. Extensive experiments on challenging datasets demonstrate our method achieves state-of-the-art performance across various metrics, surpassing existing approaches and advancing the field of 3D scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16748v2">GS-TransUNet: Integrated 2D Gaussian Splatting and Transformer UNet for Accurate Skin Lesion Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 12 pages, 7 figures, SPIE Medical Imaging 2025. 13407-1340736
    </div>
    <details class="paper-abstract">
      We can achieve fast and consistent early skin cancer detection with recent developments in computer vision and deep learning techniques. However, the existing skin lesion segmentation and classification prediction models run independently, thus missing potential efficiencies from their integrated execution. To unify skin lesion analysis, our paper presents the Gaussian Splatting - Transformer UNet (GS-TransUNet), a novel approach that synergistically combines 2D Gaussian splatting with the Transformer UNet architecture for automated skin cancer diagnosis. Our unified deep learning model efficiently delivers dual-function skin lesion classification and segmentation for clinical diagnosis. Evaluated on ISIC-2017 and PH2 datasets, our network demonstrates superior performance compared to existing state-of-the-art models across multiple metrics through 5-fold cross-validation. Our findings illustrate significant advancements in the precision of segmentation and classification. This integration sets new benchmarks in the field and highlights the potential for further research into multi-task medical image analysis methodologies, promising enhancements in automated diagnostic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14921v1">Stereo-GS: Multi-View Stereo Vision Model for Generalizable 3D Gaussian Splatting Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 ACMMM2025. Non-camera-ready version
    </div>
    <details class="paper-abstract">
      Generalizable 3D Gaussian Splatting reconstruction showcases advanced Image-to-3D content creation but requires substantial computational resources and large datasets, posing challenges to training models from scratch. Current methods usually entangle the prediction of 3D Gaussian geometry and appearance, which rely heavily on data-driven priors and result in slow regression speeds. To address this, we propose \method, a disentangled framework for efficient 3D Gaussian prediction. Our method extracts features from local image pairs using a stereo vision backbone and fuses them via global attention blocks. Dedicated point and Gaussian prediction heads generate multi-view point-maps for geometry and Gaussian features for appearance, combined as GS-maps to represent the 3DGS object. A refinement network enhances these GS-maps for high-quality reconstruction. Unlike existing methods that depend on camera parameters, our approach achieves pose-free 3D reconstruction, improving robustness and practicality. By reducing resource demands while maintaining high-quality outputs, \method provides an efficient, scalable solution for real-world 3D content generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09993v2">3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Submitted to WACV 2026
    </div>
    <details class="paper-abstract">
      Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. Existing 2D and 3D physical attacks, due to their focus on texture optimization, often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture optimization, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module that filters outliers to preserve geometric fidelity, and a physical augmentation module that simulates complex physical scenarios to enhance attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21\% to 7.38\%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16737v2">Point'n Move: Interactive Scene Object Manipulation on Gaussian Splatting Radiance Fields</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Code: https://github.com/jhuangBU/pnm
    </div>
    <details class="paper-abstract">
      We propose Point'n Move, a method that achieves interactive scene object manipulation with exposed region inpainting. Interactivity here further comes from intuitive object selection and real-time editing. To achieve this, we adopt Gaussian Splatting Radiance Field as the scene representation and fully leverage its explicit nature and speed advantage. Its explicit representation formulation allows us to devise a 2D prompt points to 3D mask dual-stage self-prompting segmentation algorithm, perform mask refinement and merging, minimize change as well as provide good initialization for scene inpainting and perform editing in real-time without per-editing training, all leads to superior quality and performance. We test our method by performing editing on both forward-facing and 360 scenes. We also compare our method against existing scene object removal methods, showing superior quality despite being more capable and having a speed advantage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14505v1">DCHM: Depth-Consistent Human Modeling for Multiview Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 multi-view detection, sparse-view reconstruction
    </div>
    <details class="paper-abstract">
      Multiview pedestrian detection typically involves two stages: human modeling and pedestrian localization. Human modeling represents pedestrians in 3D space by fusing multiview information, making its quality crucial for detection accuracy. However, existing methods often introduce noise and have low precision. While some approaches reduce noise by fitting on costly multiview 3D annotations, they often struggle to generalize across diverse scenes. To eliminate reliance on human-labeled annotations and accurately model humans, we propose Depth-Consistent Human Modeling (DCHM), a framework designed for consistent depth estimation and multiview fusion in global coordinates. Specifically, our proposed pipeline with superpixel-wise Gaussian Splatting achieves multiview depth consistency in sparse-view, large-scaled, and crowded scenarios, producing precise point clouds for pedestrian localization. Extensive validations demonstrate that our method significantly reduces noise during human modeling, outperforming previous state-of-the-art baselines. Additionally, to our knowledge, DCHM is the first to reconstruct pedestrians and perform multiview segmentation in such a challenging setting. Code is available on the \href{https://jiahao-ma.github.io/DCHM/}{project page}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14501v1">Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 A project page associated with this survey is available at https://fnzhan.com/projects/Feed-Forward-3D
    </div>
    <details class="paper-abstract">
      3D reconstruction and view synthesis are foundational problems in computer vision, graphics, and immersive technologies such as augmented reality (AR), virtual reality (VR), and digital twins. Traditional methods rely on computationally intensive iterative optimization in a complex chain, limiting their applicability in real-world scenarios. Recent advances in feed-forward approaches, driven by deep learning, have revolutionized this field by enabling fast and generalizable 3D reconstruction and view synthesis. This survey offers a comprehensive review of feed-forward techniques for 3D reconstruction and view synthesis, with a taxonomy according to the underlying representation architectures including point cloud, 3D Gaussian Splatting (3DGS), Neural Radiance Fields (NeRF), etc. We examine key tasks such as pose-free reconstruction, dynamic 3D reconstruction, and 3D-aware image and video synthesis, highlighting their applications in digital humans, SLAM, robotics, and beyond. In addition, we review commonly used datasets with detailed statistics, along with evaluation protocols for various downstream tasks. We conclude by discussing open research challenges and promising directions for future work, emphasizing the potential of feed-forward approaches to advance the state of the art in 3D vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14454v1">Adaptive 3D Gaussian Splatting Video Streaming: Visual Saliency-Aware Tiling and Meta-Learning-Based Bitrate Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting video (3DGS) streaming has recently emerged as a research hotspot in both academia and industry, owing to its impressive ability to deliver immersive 3D video experiences. However, research in this area is still in its early stages, and several fundamental challenges, such as tiling, quality assessment, and bitrate adaptation, require further investigation. In this paper, we tackle these challenges by proposing a comprehensive set of solutions. Specifically, we propose an adaptive 3DGS tiling technique guided by saliency analysis, which integrates both spatial and temporal features. Each tile is encoded into versions possessing dedicated deformation fields and multiple quality levels for adaptive selection. We also introduce a novel quality assessment framework for 3DGS video that jointly evaluates spatial-domain degradation in 3DGS representations during streaming and the quality of the resulting 2D rendered images. Additionally, we develop a meta-learning-based adaptive bitrate algorithm specifically tailored for 3DGS video streaming, achieving optimal performance across varying network conditions. Extensive experiments demonstrate that our proposed approaches significantly outperform state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14432v1">Adaptive 3D Gaussian Splatting Video Streaming</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      The advent of 3D Gaussian splatting (3DGS) has significantly enhanced the quality of volumetric video representation. Meanwhile, in contrast to conventional volumetric video, 3DGS video poses significant challenges for streaming due to its substantially larger data volume and the heightened complexity involved in compression and transmission. To address these issues, we introduce an innovative framework for 3DGS volumetric video streaming. Specifically, we design a 3DGS video construction method based on the Gaussian deformation field. By employing hybrid saliency tiling and differentiated quality modeling of 3DGS video, we achieve efficient data compression and adaptation to bandwidth fluctuations while ensuring high transmission quality. Then we build a complete 3DGS video streaming system and validate the transmission performance. Through experimental evaluation, our method demonstrated superiority over existing approaches in various aspects, including video quality, compression effectiveness, and transmission rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13917v1">Neural-GASh: A CGA-based neural radiance prediction pipeline for real-time shading</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper presents Neural-GASh, a novel real-time shading pipeline for 3D meshes, that leverages a neural radiance field architecture to perform image-based rendering (IBR) using Conformal Geometric Algebra (CGA)-encoded vertex information as input. Unlike traditional Precomputed Radiance Transfer (PRT) methods, that require expensive offline precomputations, our learned model directly consumes CGA-based representations of vertex positions and normals, enabling dynamic scene shading without precomputation. Integrated seamlessly into the Unity engine, Neural-GASh facilitates accurate shading of animated and deformed 3D meshes - capabilities essential for dynamic, interactive environments. The shading of the scene is implemented within Unity, where rotation of scene lights in terms of Spherical Harmonics is also performed optimally using CGA. This neural field approach is designed to deliver fast and efficient light transport simulation across diverse platforms, including mobile and VR, while preserving high rendering quality. Additionally, we evaluate our method on scenes generated via 3D Gaussian splats, further demonstrating the flexibility and robustness of Neural-GASh in diverse scenarios. Performance is evaluated in comparison to conventional PRT, demonstrating competitive rendering speeds even with complex geometries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13891v1">PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Accepted by ICCV2025
    </div>
    <details class="paper-abstract">
      COLMAP-free 3D Gaussian Splatting (3D-GS) has recently attracted increasing attention due to its remarkable performance in reconstructing high-quality 3D scenes from unposed images or videos. However, it often struggles to handle scenes with complex camera trajectories as featured by drastic rotation and translation across adjacent camera views, leading to degraded estimation of camera poses and further local minima in joint optimization of camera poses and 3D-GS. We propose PCR-GS, an innovative COLMAP-free 3DGS technique that achieves superior 3D scene modeling and camera pose estimation via camera pose co-regularization. PCR-GS achieves regularization from two perspectives. The first is feature reprojection regularization which extracts view-robust DINO features from adjacent camera views and aligns their semantic information for camera pose regularization. The second is wavelet-based frequency regularization which exploits discrepancy in high-frequency details to further optimize the rotation matrix in camera poses. Extensive experiments over multiple real-world scenes show that the proposed PCR-GS achieves superior pose-free 3D-GS scene modeling under dramatic changes of camera trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23957v2">GaVS: 3D-Grounded Video Stabilization via Temporally-Consistent Local Reconstruction and Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 siggraph 2025, project website: https://sinoyou.github.io/gavs. version 2, update discussion
    </div>
    <details class="paper-abstract">
      Video stabilization is pivotal for video processing, as it removes unwanted shakiness while preserving the original user motion intent. Existing approaches, depending on the domain they operate, suffer from several issues (e.g. geometric distortions, excessive cropping, poor generalization) that degrade the user experience. To address these issues, we introduce \textbf{GaVS}, a novel 3D-grounded approach that reformulates video stabilization as a temporally-consistent `local reconstruction and rendering' paradigm. Given 3D camera poses, we augment a reconstruction model to predict Gaussian Splatting primitives, and finetune it at test-time, with multi-view dynamics-aware photometric supervision and cross-frame regularization, to produce temporally-consistent local reconstructions. The model are then used to render each stabilized frame. We utilize a scene extrapolation module to avoid frame cropping. Our method is evaluated on a repurposed dataset, instilled with 3D-grounded information, covering samples with diverse camera motions and scene dynamics. Quantitatively, our method is competitive with or superior to state-of-the-art 2D and 2.5D approaches in terms of conventional task metrics and new geometry consistency. Qualitatively, our method produces noticeably better results compared to alternatives, validated by the user study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13586v1">TexGS-VolVis: Expressive Scene Editing for Volume Visualization via Textured Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Accepted by IEEE VIS 2025
    </div>
    <details class="paper-abstract">
      Advancements in volume visualization (VolVis) focus on extracting insights from 3D volumetric data by generating visually compelling renderings that reveal complex internal structures. Existing VolVis approaches have explored non-photorealistic rendering techniques to enhance the clarity, expressiveness, and informativeness of visual communication. While effective, these methods often rely on complex predefined rules and are limited to transferring a single style, restricting their flexibility. To overcome these limitations, we advocate the representation of VolVis scenes using differentiable Gaussian primitives combined with pretrained large models to enable arbitrary style transfer and real-time rendering. However, conventional 3D Gaussian primitives tightly couple geometry and appearance, leading to suboptimal stylization results. To address this, we introduce TexGS-VolVis, a textured Gaussian splatting framework for VolVis. TexGS-VolVis employs 2D Gaussian primitives, extending each Gaussian with additional texture and shading attributes, resulting in higher-quality, geometry-consistent stylization and enhanced lighting control during inference. Despite these improvements, achieving flexible and controllable scene editing remains challenging. To further enhance stylization, we develop image- and text-driven non-photorealistic scene editing tailored for TexGS-VolVis and 2D-lift-3D segmentation to enable partial editing with fine-grained control. We evaluate TexGS-VolVis both qualitatively and quantitatively across various volume rendering scenes, demonstrating its superiority over existing methods in terms of efficiency, visual quality, and editing flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08136v2">RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated its potential in reconstructing scenes from unposed images. However, optimization-based 3DGS methods struggle with sparse views due to limited prior knowledge. Meanwhile, feed-forward Gaussian approaches are constrained by input formats, making it challenging to incorporate more input views. To address these challenges, we propose RegGS, a 3D Gaussian registration-based framework for reconstructing unposed sparse views. RegGS aligns local 3D Gaussians generated by a feed-forward network into a globally consistent 3D Gaussian representation. Technically, we implement an entropy-regularized Sinkhorn algorithm to efficiently solve the optimal transport Mixture 2-Wasserstein $(\text{MW}_2)$ distance, which serves as an alignment metric for Gaussian mixture models (GMMs) in $\mathrm{Sim}(3)$ space. Furthermore, we design a joint 3DGS registration module that integrates the $\text{MW}_2$ distance, photometric consistency, and depth geometry. This enables a coarse-to-fine registration process while accurately estimating camera poses and aligning the scene. Experiments on the RE10K and ACID datasets demonstrate that RegGS effectively registers local Gaussians with high fidelity, achieving precise pose estimation and high-quality novel-view synthesis. Project page: https://3dagentworld.github.io/reggs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11069v2">TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a {\delta} < 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images. Code and more results are available at https://jeongyun0609.github.io/TRAN-D/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12335v3">GS-I$^{3}$: Gaussian Splatting for Surface Reconstruction from Illumination-Inconsistent Images</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Accurate geometric surface reconstruction, providing essential environmental information for navigation and manipulation tasks, is critical for enabling robotic self-exploration and interaction. Recently, 3D Gaussian Splatting (3DGS) has gained significant attention in the field of surface reconstruction due to its impressive geometric quality and computational efficiency. While recent relevant advancements in novel view synthesis under inconsistent illumination using 3DGS have shown promise, the challenge of robust surface reconstruction under such conditions is still being explored. To address this challenge, we propose a method called GS-3I. Specifically, to mitigate 3D Gaussian optimization bias caused by underexposed regions in single-view images, based on Convolutional Neural Network (CNN), a tone mapping correction framework is introduced. Furthermore, inconsistent lighting across multi-view images, resulting from variations in camera settings and complex scene illumination, often leads to geometric constraint mismatches and deviations in the reconstructed surface. To overcome this, we propose a normal compensation mechanism that integrates reference normals extracted from single-view image with normals computed from multi-view observations to effectively constrain geometric inconsistencies. Extensive experimental evaluations demonstrate that GS-3I can achieve robust and accurate surface reconstruction across complex illumination scenarios, highlighting its effectiveness and versatility in this critical challenge. https://github.com/TFwang-9527/GS-3I
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12137v1">AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Modeling and rendering dynamic urban driving scenes is crucial for self-driving simulation. Current high-quality methods typically rely on costly manual object tracklet annotations, while self-supervised approaches fail to capture dynamic object motions accurately and decompose scenes properly, resulting in rendering artifacts. We introduce AD-GS, a novel self-supervised framework for high-quality free-viewpoint rendering of driving scenes from a single log. At its core is a novel learnable motion model that integrates locality-aware B-spline curves with global-aware trigonometric functions, enabling flexible yet precise dynamic object modeling. Rather than requiring comprehensive semantic labeling, AD-GS automatically segments scenes into objects and background with the simplified pseudo 2D segmentation, representing objects using dynamic Gaussians and bidirectional temporal visibility masks. Further, our model incorporates visibility reasoning and physically rigid regularization to enhance robustness. Extensive evaluations demonstrate that our annotation-free model significantly outperforms current state-of-the-art annotation-free methods and is competitive with annotation-dependent approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12095v1">BRUM: Robust 3D Vehicle Reconstruction from 360 Sparse Images</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Accurate 3D reconstruction of vehicles is vital for applications such as vehicle inspection, predictive maintenance, and urban planning. Existing methods like Neural Radiance Fields and Gaussian Splatting have shown impressive results but remain limited by their reliance on dense input views, which hinders real-world applicability. This paper addresses the challenge of reconstructing vehicles from sparse-view inputs, leveraging depth maps and a robust pose estimation architecture to synthesize novel views and augment training data. Specifically, we enhance Gaussian Splatting by integrating a selective photometric loss, applied only to high-confidence pixels, and replacing standard Structure-from-Motion pipelines with the DUSt3R architecture to improve camera pose estimation. Furthermore, we present a novel dataset featuring both synthetic and real-world public transportation vehicles, enabling extensive evaluation of our approach. Experimental results demonstrate state-of-the-art performance across multiple benchmarks, showcasing the method's ability to achieve high-quality reconstructions even under constrained input conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05935v2">SurGSplat: Progressive Geometry-Constrained Gaussian Splatting for Surgical Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Intraoperative navigation relies heavily on precise 3D reconstruction to ensure accuracy and safety during surgical procedures. However, endoscopic scenarios present unique challenges, including sparse features and inconsistent lighting, which render many existing Structure-from-Motion (SfM)-based methods inadequate and prone to reconstruction failure. To mitigate these constraints, we propose SurGSplat, a novel paradigm designed to progressively refine 3D Gaussian Splatting (3DGS) through the integration of geometric constraints. By enabling the detailed reconstruction of vascular structures and other critical features, SurGSplat provides surgeons with enhanced visual clarity, facilitating precise intraoperative decision-making. Experimental evaluations demonstrate that SurGSplat achieves superior performance in both novel view synthesis (NVS) and pose estimation accuracy, establishing it as a high-fidelity and efficient solution for surgical scene reconstruction. More information and results can be found on the page https://surgsplat.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12027v1">SGLoc: Semantic Localization System for Camera Pose Estimation from 3D Gaussian Splatting Representation</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 8 pages, 2 figures, IROS 2025
    </div>
    <details class="paper-abstract">
      We propose SGLoc, a novel localization system that directly regresses camera poses from 3D Gaussian Splatting (3DGS) representation by leveraging semantic information. Our method utilizes the semantic relationship between 2D image and 3D scene representation to estimate the 6DoF pose without prior pose information. In this system, we introduce a multi-level pose regression strategy that progressively estimates and refines the pose of query image from the global 3DGS map, without requiring initial pose priors. Moreover, we introduce a semantic-based global retrieval algorithm that establishes correspondences between 2D (image) and 3D (3DGS map). By matching the extracted scene semantic descriptors of 2D query image and 3DGS semantic representation, we align the image with the local region of the global 3DGS map, thereby obtaining a coarse pose estimation. Subsequently, we refine the coarse pose by iteratively optimizing the difference between the query image and the rendered image from 3DGS. Our SGLoc demonstrates superior performance over baselines on 12scenes and 7scenes datasets, showing excellent capabilities in global localization without initial pose prior. Code will be available at https://github.com/IRMVLab/SGLoc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11931v1">Dark-EvGS: Event Camera as an Eye for Radiance Field in the Dark</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      In low-light environments, conventional cameras often struggle to capture clear multi-view images of objects due to dynamic range limitations and motion blur caused by long exposure. Event cameras, with their high-dynamic range and high-speed properties, have the potential to mitigate these issues. Additionally, 3D Gaussian Splatting (GS) enables radiance field reconstruction, facilitating bright frame synthesis from multiple viewpoints in low-light conditions. However, naively using an event-assisted 3D GS approach still faced challenges because, in low light, events are noisy, frames lack quality, and the color tone may be inconsistent. To address these issues, we propose Dark-EvGS, the first event-assisted 3D GS framework that enables the reconstruction of bright frames from arbitrary viewpoints along the camera trajectory. Triplet-level supervision is proposed to gain holistic knowledge, granular details, and sharp scene rendering. The color tone matching block is proposed to guarantee the color consistency of the rendered frames. Furthermore, we introduce the first real-captured dataset for the event-guided bright frame synthesis task via 3D GS-based radiance field reconstruction. Experiments demonstrate that our method achieves better results than existing methods, conquering radiance field reconstruction under challenging low-light conditions. The code and sample data are included in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12667v1">VolSegGS: Segmentation and Tracking in Dynamic Volumetric Scenes via Deformable 3D Gaussians</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Visualization of large-scale time-dependent simulation data is crucial for domain scientists to analyze complex phenomena, but it demands significant I/O bandwidth, storage, and computational resources. To enable effective visualization on local, low-end machines, recent advances in view synthesis techniques, such as neural radiance fields, utilize neural networks to generate novel visualizations for volumetric scenes. However, these methods focus on reconstruction quality rather than facilitating interactive visualization exploration, such as feature extraction and tracking. We introduce VolSegGS, a novel Gaussian splatting framework that supports interactive segmentation and tracking in dynamic volumetric scenes for exploratory visualization and analysis. Our approach utilizes deformable 3D Gaussians to represent a dynamic volumetric scene, allowing for real-time novel view synthesis. For accurate segmentation, we leverage the view-independent colors of Gaussians for coarse-level segmentation and refine the results with an affinity field network for fine-level segmentation. Additionally, by embedding segmentation results within the Gaussians, we ensure that their deformation enables continuous tracking of segmented regions over time. We demonstrate the effectiveness of VolSegGS with several time-varying datasets and compare our solutions against state-of-the-art methods. With the ability to interact with a dynamic scene in real time and provide flexible segmentation and tracking capabilities, VolSegGS offers a powerful solution under low computational demands. This framework unlocks exciting new possibilities for time-varying volumetric data analysis and visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12621v1">NLI4VolVis: Natural Language Interaction for Volume Visualization via LLM Multi-Agents and Editable 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 IEEE VIS 2025. Project Page: https://nli4volvis.github.io/
    </div>
    <details class="paper-abstract">
      Traditional volume visualization (VolVis) methods, like direct volume rendering, suffer from rigid transfer function designs and high computational costs. Although novel view synthesis approaches enhance rendering efficiency, they require additional learning effort for non-experts and lack support for semantic-level interaction. To bridge this gap, we propose NLI4VolVis, an interactive system that enables users to explore, query, and edit volumetric scenes using natural language. NLI4VolVis integrates multi-view semantic segmentation and vision-language models to extract and understand semantic components in a scene. We introduce a multi-agent large language model architecture equipped with extensive function-calling tools to interpret user intents and execute visualization tasks. The agents leverage external tools and declarative VolVis commands to interact with the VolVis engine powered by 3D editable Gaussians, enabling open-vocabulary object querying, real-time scene editing, best-view selection, and 2D stylization. We validate our system through case studies and a user study, highlighting its improved accessibility and usability in volumetric data exploration. We strongly recommend readers check our case studies, demo video, and source code at https://nli4volvis.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12498v1">Wavelet-GS: 3D Gaussian Splatting with Wavelet Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has revolutionized 3D scene reconstruction, which effectively balances rendering quality, efficiency, and speed. However, existing 3DGS approaches usually generate plausible outputs and face significant challenges in complex scene reconstruction, manifesting as incomplete holistic structural outlines and unclear local lighting effects. To address these issues simultaneously, we propose a novel decoupled optimization framework, which integrates wavelet decomposition into 3D Gaussian Splatting and 2D sampling. Technically, through 3D wavelet decomposition, our approach divides point clouds into high-frequency and low-frequency components, enabling targeted optimization for each. The low-frequency component captures global structural outlines and manages the distribution of Gaussians through voxelization. In contrast, the high-frequency component restores intricate geometric and textural details while incorporating a relight module to mitigate lighting artifacts and enhance photorealistic rendering. Additionally, a 2D wavelet decomposition is applied to the training images, simulating radiance variations. This provides critical guidance for high-frequency detail reconstruction, ensuring seamless integration of details with the global structure. Extensive experiments on challenging datasets demonstrate our method achieves state-of-the-art performance across various metrics, surpassing existing approaches and advancing the field of 3D scene reconstruction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11321v1">A Mixed-Primitive-based Gaussian Splatting Method for Surface Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Recently, Gaussian Splatting (GS) has received a lot of attention in surface reconstruction. However, while 3D objects can be of complex and diverse shapes in the real world, existing GS-based methods only limitedly use a single type of splatting primitive (Gaussian ellipse or Gaussian ellipsoid) to represent object surfaces during their reconstruction. In this paper, we highlight that this can be insufficient for object surfaces to be represented in high quality. Thus, we propose a novel framework that, for the first time, enables Gaussian Splatting to incorporate multiple types of (geometrical) primitives during its surface reconstruction process. Specifically, in our framework, we first propose a compositional splatting strategy, enabling the splatting and rendering of different types of primitives in the Gaussian Splatting pipeline. In addition, we also design our framework with a mixed-primitive-based initialization strategy and a vertex pruning mechanism to further promote its surface representation learning process to be well executed leveraging different types of primitives. Extensive experiments show the efficacy of our framework and its accurate surface reconstruction performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11069v1">TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a {\delta} < 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images. Code and more results are available at https://jeongyun0609.github.io/TRAN-D/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11061v1">Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Recent advances in 3D neural representations and instance-level editing models have enabled the efficient creation of high-quality 3D content. However, achieving precise local 3D edits remains challenging, especially for Gaussian Splatting, due to inconsistent multi-view 2D part segmentations and inherently ambiguous nature of Score Distillation Sampling (SDS) loss. To address these limitations, we propose RoMaP, a novel local 3D Gaussian editing framework that enables precise and drastic part-level modifications. First, we introduce a robust 3D mask generation module with our 3D-Geometry Aware Label Prediction (3D-GALP), which uses spherical harmonics (SH) coefficients to model view-dependent label variations and soft-label property, yielding accurate and consistent part segmentations across viewpoints. Second, we propose a regularized SDS loss that combines the standard SDS loss with additional regularizers. In particular, an L1 anchor loss is introduced via our Scheduled Latent Mixing and Part (SLaMP) editing method, which generates high-quality part-edited 2D images and confines modifications only to the target region while preserving contextual coherence. Additional regularizers, such as Gaussian prior removal, further improve flexibility by allowing changes beyond the existing context, and robust 3D masking prevents unintended edits. Experimental results demonstrate that our RoMaP achieves state-of-the-art local 3D editing on both reconstructed and generated Gaussian scenes and objects qualitatively and quantitatively, making it possible for more robust and flexible part-level 3D Gaussian editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10542v1">ScaffoldAvatar: High-Fidelity Gaussian Avatars with Patch Expressions</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 (SIGGRAPH 2025) Paper Video: https://youtu.be/VyWkgsGdbkk Project Page: https://shivangi-aneja.github.io/projects/scaffoldavatar/
    </div>
    <details class="paper-abstract">
      Generating high-fidelity real-time animated sequences of photorealistic 3D head avatars is important for many graphics applications, including immersive telepresence and movies. This is a challenging problem particularly when rendering digital avatar close-ups for showing character's facial microfeatures and expressions. To capture the expressive, detailed nature of human heads, including skin furrowing and finer-scale facial movements, we propose to couple locally-defined facial expressions with 3D Gaussian splatting to enable creating ultra-high fidelity, expressive and photorealistic 3D head avatars. In contrast to previous works that operate on a global expression space, we condition our avatar's dynamics on patch-based local expression features and synthesize 3D Gaussians at a patch level. In particular, we leverage a patch-based geometric 3D face model to extract patch expressions and learn how to translate these into local dynamic skin appearance and motion by coupling the patches with anchor points of Scaffold-GS, a recent hierarchical scene representation. These anchors are then used to synthesize 3D Gaussians on-the-fly, conditioned by patch-expressions and viewing direction. We employ color-based densification and progressive training to obtain high-quality results and faster convergence for high resolution 3K training images. By leveraging patch-level expressions, ScaffoldAvatar consistently achieves state-of-the-art performance with visually natural motion, while encompassing diverse facial expressions and styles in real time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11447v4">GaussianOcc: Fully Self-supervised and Efficient 3D Occupancy Estimation with Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Project page: https://ganwanshui.github.io/GaussianOcc/
    </div>
    <details class="paper-abstract">
      We introduce GaussianOcc, a systematic method that investigates the two usages of Gaussian splatting for fully self-supervised and efficient 3D occupancy estimation in surround views. First, traditional methods for self-supervised 3D occupancy estimation still require ground truth 6D poses from sensors during training. To address this limitation, we propose Gaussian Splatting for Projection (GSP) module to provide accurate scale information for fully self-supervised training from adjacent view projection. Additionally, existing methods rely on volume rendering for final 3D voxel representation learning using 2D signals (depth maps, semantic maps), which is both time-consuming and less effective. We propose Gaussian Splatting from Voxel space (GSV) to leverage the fast rendering properties of Gaussian splatting. As a result, the proposed GaussianOcc method enables fully self-supervised (no ground truth pose) 3D occupancy estimation in competitive performance with low computational cost (2.7 times faster in training and 5 times faster in rendering). The relevant code is available in https://github.com/GANWANSHUI/GaussianOcc.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06269v2">BayesSDF: Surface-Based Laplacian Uncertainty Estimation for 3D Geometry with Neural Signed Distance Fields</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 ICCV 2025 Workshops (8 Pages, 6 Figures, 2 Tables)
    </div>
    <details class="paper-abstract">
      Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and reliable uncertainty estimates are essential. Unlike radiance-based models such as Neural Radiance Fields (NeRF) or 3D Gaussian splatting, which lack explicit surface formulations, Signed Distance Functions (SDFs) define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability using Hessian-based metrics, enabling efficient, surfaceaware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08331v2">SLGaussian: Fast Language Gaussian Splatting in Sparse Views</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted by ACM MM 2025. Project page: https://chenkangjie1123.github.io/SLGaussian.github.io/
    </div>
    <details class="paper-abstract">
      3D semantic field learning is crucial for applications like autonomous navigation, AR/VR, and robotics, where accurate comprehension of 3D scenes from limited viewpoints is essential. Existing methods struggle under sparse view conditions, relying on inefficient per-scene multi-view optimizations, which are impractical for many real-world tasks. To address this, we propose SLGaussian, a feed-forward method for constructing 3D semantic fields from sparse viewpoints, allowing direct inference of 3DGS-based scenes. By ensuring consistent SAM segmentations through video tracking and using low-dimensional indexing for high-dimensional CLIP features, SLGaussian efficiently embeds language information in 3D space, offering a robust solution for accurate 3D scene understanding under sparse view conditions. In experiments on two-view sparse 3D object querying and segmentation in the LERF and 3D-OVS datasets, SLGaussian outperforms existing methods in chosen IoU, Localization Accuracy, and mIoU. Moreover, our model achieves scene inference in under 30 seconds and open-vocabulary querying in just 0.011 seconds per query.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05332v2">CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Revised Version of CRiM-GS, Project Page: https://Jho-Yonsei.github.io/CoMoGaussian
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has gained significant attention due to its high-quality novel view rendering, motivating research to address real-world challenges. A critical issue is the camera motion blur caused by movement during exposure, which hinders accurate 3D scene reconstruction. In this study, we propose CoMoGaussian, a Continuous Motion-Aware Gaussian Splatting that reconstructs precise 3D scenes from motion-blurred images while maintaining real-time rendering speed. Considering the complex motion patterns inherent in real-world camera movements, we predict continuous camera trajectories using neural ordinary differential equations (ODEs). To ensure accurate modeling, we employ rigid body transformations, preserving the shape and size of the object but rely on the discrete integration of sampled frames. To better approximate the continuous nature of motion blur, we introduce a continuous motion refinement (CMR) transformation that refines rigid transformations by incorporating additional learnable parameters. By revisiting fundamental camera theory and leveraging advanced neural ODE techniques, we achieve precise modeling of continuous camera trajectories, leading to improved reconstruction accuracy. Extensive experiments demonstrate state-of-the-art performance both quantitatively and qualitatively on benchmark datasets, which include a wide range of motion blur scenarios, from moderate to extreme blur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02178v2">Sparfels: Fast Reconstruction from Sparse Unposed Imagery</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 ICCV 2025. Project page : https://shubhendu-jena.github.io/Sparfels-web/
    </div>
    <details class="paper-abstract">
      We present a method for Sparse view reconstruction with surface element splatting that runs within 3 minutes on a consumer grade GPU. While few methods address sparse radiance field learning from noisy or unposed sparse cameras, shape recovery remains relatively underexplored in this setting. Several radiance and shape learning test-time optimization methods address the sparse posed setting by learning data priors or using combinations of external monocular geometry priors. Differently, we propose an efficient and simple pipeline harnessing a single recent 3D foundation model. We leverage its various task heads, notably point maps and camera initializations to instantiate a bundle adjusting 2D Gaussian Splatting (2DGS) model, and image correspondences to guide camera optimization midst 2DGS training. Key to our contribution is a novel formulation of splatted color variance along rays, which can be computed efficiently. Reducing this moment in training leads to more accurate shape reconstructions. We demonstrate state-of-the-art performances in the sparse uncalibrated setting in reconstruction and novel view benchmarks based on established multi-view datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09993v1">3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Submitted to WACV 2026
    </div>
    <details class="paper-abstract">
      Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. While existing 2D and 3D physical attacks typically optimize texture, they often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module to preserve geometric fidelity, and a physical augmentation module to simulate complex physical scenarios, thus enhancing attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21% to 7.38%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks. These results validate 3DGAA as a practical attack framework for evaluating the safety of perception systems in autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13613v2">MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      4D Gaussian Splatting (4DGS) has recently emerged as a promising technique for capturing complex dynamic 3D scenes with high fidelity. It utilizes a 4D Gaussian representation and a GPU-friendly rasterizer, enabling rapid rendering speeds. Despite its advantages, 4DGS faces significant challenges, notably the requirement of millions of 4D Gaussians, each with extensive associated attributes, leading to substantial memory and storage cost. This paper introduces a memory-efficient framework for 4DGS. We streamline the color attribute by decomposing it into a per-Gaussian direct color component with only 3 parameters and a shared lightweight alternating current color predictor. This approach eliminates the need for spherical harmonics coefficients, which typically involve up to 144 parameters in classic 4DGS, thereby creating a memory-efficient 4D Gaussian representation. Furthermore, we introduce an entropy-constrained Gaussian deformation technique that uses a deformation field to expand the action range of each Gaussian and integrates an opacity-based entropy loss to limit the number of Gaussians, thus forcing our model to use as few Gaussians as possible to fit a dynamic scene well. With simple half-precision storage and zip compression, our framework achieves a storage reduction by approximately 190$\times$ and 125$\times$ on the Technicolor and Neural 3D Video datasets, respectively, compared to the original 4DGS. Meanwhile, it maintains comparable rendering speeds and scene representation quality, setting a new standard in the field. Code is available at https://github.com/Xinjie-Q/MEGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05242v3">SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding</a></div>
    <div class="paper-meta">
      📅 2025-07-13
      | 💬 ICCV 2025 accept;code, video, demos, and project are available at Project page https://segs-slam.github.io/
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3D-GS) has recently revolutionized novel view synthesis in the simultaneous localization and mapping (SLAM) problem. However, most existing algorithms fail to fully capture the underlying structure, resulting in structural inconsistency. Additionally, they struggle with abrupt appearance variations, leading to inconsistent visual quality. To address these problems, we propose SEGS-SLAM, a structure-enhanced 3D Gaussian Splatting SLAM, which achieves high-quality photorealistic mapping. Our main contributions are two-fold. First, we propose a structure-enhanced photorealistic mapping (SEPM) framework that, for the first time, leverages highly structured point cloud to initialize structured 3D Gaussians, leading to significant improvements in rendering quality. Second, we propose Appearance-from-Motion embedding (AfME), enabling 3D Gaussians to better model image appearance variations across different camera poses. Extensive experiments on monocular, stereo, and RGB-D datasets demonstrate that SEGS-SLAM significantly outperforms state-of-the-art (SOTA) methods in photorealistic mapping quality, e.g., an improvement of $19.86\%$ in PSNR over MonoGS on the TUM RGB-D dataset for monocular cameras. The project page is available at https://segs-slam.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08726v1">Learning human-to-robot handovers through 3D scene reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 8 pages, 6 figures, 2 table
    </div>
    <details class="paper-abstract">
      Learning robot manipulation policies from raw, real-world image data requires a large number of robot-action trials in the physical environment. Although training using simulations offers a cost-effective alternative, the visual domain gap between simulation and robot workspace remains a major limitation. Gaussian Splatting visual reconstruction methods have recently provided new directions for robot manipulation by generating realistic environments. In this paper, we propose the first method for learning supervised-based robot handovers solely from RGB images without the need of real-robot training or real-robot data collection. The proposed policy learner, Human-to-Robot Handover using Sparse-View Gaussian Splatting (H2RH-SGS), leverages sparse-view Gaussian Splatting reconstruction of human-to-robot handover scenes to generate robot demonstrations containing image-action pairs captured with a camera mounted on the robot gripper. As a result, the simulated camera pose changes in the reconstructed scene can be directly translated into gripper pose changes. We train a robot policy on demonstrations collected with 16 household objects and {\em directly} deploy this policy in the real environment. Experiments in both Gaussian Splatting reconstructed scene and real-world human-to-robot handover experiments demonstrate that H2RH-SGS serves as a new and effective representation for the human-to-robot handover task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11868v3">MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Accurately analyzing the motion parts and their motion attributes in dynamic environments is crucial for advancing key areas such as embodied intelligence. Addressing the limitations of existing methods that rely on dense multi-view images or detailed part-level annotations, we propose an innovative framework that can analyze 3D mobility from monocular videos in a zero-shot manner. This framework can precisely parse motion parts and motion attributes only using a monocular video, completely eliminating the need for annotated training data. Specifically, our method first constructs the scene geometry and roughly analyzes the motion parts and their initial motion attributes combining depth estimation, optical flow analysis and point cloud registration method, then employs 2D Gaussian splatting for scene representation. Building on this, we introduce an end-to-end dynamic scene optimization algorithm specifically designed for articulated objects, refining the initial analysis results to ensure the system can handle 'rotation', 'translation', and even complex movements ('rotation+translation'), demonstrating high flexibility and versatility. To validate the robustness and wide applicability of our method, we created a comprehensive dataset comprising both simulated and real-world scenarios. Experimental results show that our framework can effectively analyze articulated object motions in an annotation-free manner, showcasing its significant potential in future embodied intelligence applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10012v2">EBAD-Gaussian: Event-driven Bundle Adjusted Deblur Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      While 3D Gaussian Splatting (3D-GS) achieves photorealistic novel view synthesis, its performance degrades with motion blur. In scenarios with rapid motion or low-light conditions, existing RGB-based deblurring methods struggle to model camera pose and radiance changes during exposure, reducing reconstruction accuracy. Event cameras, capturing continuous brightness changes during exposure, can effectively assist in modeling motion blur and improving reconstruction quality. Therefore, we propose Event-driven Bundle Adjusted Deblur Gaussian Splatting (EBAD-Gaussian), which reconstructs sharp 3D Gaussians from event streams and severely blurred images. This method jointly learns the parameters of these Gaussians while recovering camera motion trajectories during exposure time. Specifically, we first construct a blur loss function by synthesizing multiple latent sharp images during the exposure time, minimizing the difference between real and synthesized blurred images. Then we use event stream to supervise the light intensity changes between latent sharp images at any time within the exposure period, supplementing the light intensity dynamic changes lost in RGB images. Furthermore, we optimize the latent sharp images at intermediate exposure times based on the event-based double integral (EDI) prior, applying consistency constraints to enhance the details and texture information of the reconstructed images. Extensive experiments on synthetic and real-world datasets show that EBAD-Gaussian can achieve high-quality 3D scene reconstruction under the condition of blurred images and event stream inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08434v1">RePaintGS: Reference-Guided Gaussian Splatting for Realistic and View-Consistent 3D Scene Inpainting</a></div>
    <div class="paper-meta">
      📅 2025-07-11
    </div>
    <details class="paper-abstract">
      Radiance field methods, such as Neural Radiance Field or 3D Gaussian Splatting, have emerged as seminal 3D representations for synthesizing realistic novel views. For practical applications, there is ongoing research on flexible scene editing techniques, among which object removal is a representative task. However, removing objects exposes occluded regions, often leading to unnatural appearances. Thus, studies have employed image inpainting techniques to replace such regions with plausible content - a task referred to as 3D scene inpainting. However, image inpainting methods produce one of many plausible completions for each view, leading to inconsistencies between viewpoints. A widely adopted approach leverages perceptual cues to blend inpainted views smoothly. However, it is prone to detail loss and can fail when there are perceptual inconsistencies across views. In this paper, we propose a novel 3D scene inpainting method that reliably produces realistic and perceptually consistent results even for complex scenes by leveraging a reference view. Given the inpainted reference view, we estimate the inpainting similarity of the other views to adjust their contribution in constructing an accurate geometry tailored to the reference. This geometry is then used to warp the reference inpainting to other views as pseudo-ground truth, guiding the optimization to match the reference appearance. Comparative evaluation studies have shown that our approach improves both the geometric fidelity and appearance consistency of inpainted scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24204v3">GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-11
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Recent 3D Gaussian Splatting (3DGS) representations have demonstrated remarkable performance in novel view synthesis; further, material-lighting disentanglement on 3DGS warrants relighting capabilities and its adaptability to broader applications. While the general approach to the latter operation lies in integrating differentiable physically-based rendering (PBR) techniques to jointly recover BRDF materials and environment lighting, achieving a precise disentanglement remains an inherently difficult task due to the challenge of accurately modeling light transport. Existing approaches typically approximate Gaussian points' normals, which constitute an implicit geometric constraint. However, they usually suffer from inaccuracies in normal estimation that subsequently degrade light transport, resulting in noisy material decomposition and flawed relighting results. To address this, we propose GeoSplatting, a novel approach that augments 3DGS with explicit geometry guidance for precise light transport modeling. By differentiably constructing a surface-grounded 3DGS from an optimizable mesh, our approach leverages well-defined mesh normals and the opaque mesh surface, and additionally facilitates the use of mesh-based ray tracing techniques for efficient, occlusion-aware light transport calculations. This enhancement ensures precise material decomposition while preserving the efficiency and high-quality rendering capabilities of 3DGS. Comprehensive evaluations across diverse datasets demonstrate the effectiveness of GeoSplatting, highlighting its superior efficiency and state-of-the-art inverse rendering performance. The project page can be found at https://pku-vcl-geometry.github.io/GeoSplatting/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07733v1">RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities in novel view synthesis. However, rendering reflective objects remains a significant challenge, particularly in inverse rendering and relighting. We introduce RTR-GS, a novel inverse rendering framework capable of robustly rendering objects with arbitrary reflectance properties, decomposing BRDF and lighting, and delivering credible relighting results. Given a collection of multi-view images, our method effectively recovers geometric structure through a hybrid rendering model that combines forward rendering for radiance transfer with deferred rendering for reflections. This approach successfully separates high-frequency and low-frequency appearances, mitigating floating artifacts caused by spherical harmonic overfitting when handling high-frequency details. We further refine BRDF and lighting decomposition using an additional physically-based deferred rendering branch. Experimental results show that our method enhances novel view synthesis, normal estimation, decomposition, and relighting while maintaining efficient training inference process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07519v1">MUVOD: A Novel Multi-view Video Object Segmentation Dataset and A Benchmark for 3D Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      The application of methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3D GS) have steadily gained popularity in the field of 3D object segmentation in static scenes. These approaches demonstrate efficacy in a range of 3D scene understanding and editing tasks. Nevertheless, the 4D object segmentation of dynamic scenes remains an underexplored field due to the absence of a sufficiently extensive and accurately labelled multi-view video dataset. In this paper, we present MUVOD, a new multi-view video dataset for training and evaluating object segmentation in reconstructed real-world scenarios. The 17 selected scenes, describing various indoor or outdoor activities, are collected from different sources of datasets originating from various types of camera rigs. Each scene contains a minimum of 9 views and a maximum of 46 views. We provide 7830 RGB images (30 frames per video) with their corresponding segmentation mask in 4D motion, meaning that any object of interest in the scene could be tracked across temporal frames of a given view or across different views belonging to the same camera rig. This dataset, which contains 459 instances of 73 categories, is intended as a basic benchmark for the evaluation of multi-view video segmentation methods. We also present an evaluation metric and a baseline segmentation approach to encourage and evaluate progress in this evolving field. Additionally, we propose a new benchmark for 3D object segmentation task with a subset of annotated multi-view images selected from our MUVOD dataset. This subset contains 50 objects of different conditions in different scenarios, providing a more comprehensive analysis of state-of-the-art 3D object segmentation methods. Our proposed MUVOD dataset is available at https://volumetric-repository.labs.b-com.com/#/muvod.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07465v1">SD-GS: Structured Deformable 3D Gaussians for Efficient Dynamic Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Current 4D Gaussian frameworks for dynamic scene reconstruction deliver impressive visual fidelity and rendering speed, however, the inherent trade-off between storage costs and the ability to characterize complex physical motions significantly limits the practical application of these methods. To tackle these problems, we propose SD-GS, a compact and efficient dynamic Gaussian splatting framework for complex dynamic scene reconstruction, featuring two key contributions. First, we introduce a deformable anchor grid, a hierarchical and memory-efficient scene representation where each anchor point derives multiple 3D Gaussians in its local spatiotemporal region and serves as the geometric backbone of the 3D scene. Second, to enhance modeling capability for complex motions, we present a deformation-aware densification strategy that adaptively grows anchors in under-reconstructed high-dynamic regions while reducing redundancy in static areas, achieving superior visual quality with fewer anchors. Experimental results demonstrate that, compared to state-of-the-art methods, SD-GS achieves an average of 60\% reduction in model size and an average of 100\% improvement in FPS, significantly enhancing computational efficiency while maintaining or even surpassing visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07395v1">Seg-Wild: Interactive Segmentation based on 3D Gaussian Splatting for Unconstrained Image Collections</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Reconstructing and segmenting scenes from unconstrained photo collections obtained from the Internet is a novel but challenging task. Unconstrained photo collections are easier to get than well-captured photo collections. These unconstrained images suffer from inconsistent lighting and transient occlusions, which makes segmentation challenging. Previous segmentation methods cannot address transient occlusions or accurately restore the scene's lighting conditions. Therefore, we propose Seg-Wild, an interactive segmentation method based on 3D Gaussian Splatting for unconstrained image collections, suitable for in-the-wild scenes. We integrate multi-dimensional feature embeddings for each 3D Gaussian and calculate the feature similarity between the feature embeddings and the segmentation target to achieve interactive segmentation in the 3D scene. Additionally, we introduce the Spiky 3D Gaussian Cutter (SGC) to smooth abnormal 3D Gaussians. We project the 3D Gaussians onto a 2D plane and calculate the ratio of 3D Gaussians that need to be cut using the SAM mask. We also designed a benchmark to evaluate segmentation quality in in-the-wild scenes. Experimental results demonstrate that compared to previous methods, Seg-Wild achieves better segmentation results and reconstruction quality. Our code will be available at https://github.com/Sugar0725/Seg-Wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08137v1">Temporally Consistent Amodal Completion for 3D Human-Object Interaction Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for reconstructing dynamic human-object interactions from monocular video that overcomes challenges associated with occlusions and temporal inconsistencies. Traditional 3D reconstruction methods typically assume static objects or full visibility of dynamic subjects, leading to degraded performance when these assumptions are violated-particularly in scenarios where mutual occlusions occur. To address this, our framework leverages amodal completion to infer the complete structure of partially obscured regions. Unlike conventional approaches that operate on individual frames, our method integrates temporal context, enforcing coherence across video sequences to incrementally refine and stabilize reconstructions. This template-free strategy adapts to varying conditions without relying on predefined models, significantly enhancing the recovery of intricate details in dynamic scenes. We validate our approach using 3D Gaussian Splatting on challenging monocular videos, demonstrating superior precision in handling occlusions and maintaining temporal stability compared to existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08136v1">RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has demonstrated its potential in reconstructing scenes from unposed images. However, optimization-based 3DGS methods struggle with sparse views due to limited prior knowledge. Meanwhile, feed-forward Gaussian approaches are constrained by input formats, making it challenging to incorporate more input views. To address these challenges, we propose RegGS, a 3D Gaussian registration-based framework for reconstructing unposed sparse views. RegGS aligns local 3D Gaussians generated by a feed-forward network into a globally consistent 3D Gaussian representation. Technically, we implement an entropy-regularized Sinkhorn algorithm to efficiently solve the optimal transport Mixture 2-Wasserstein $(\text{MW}_2)$ distance, which serves as an alignment metric for Gaussian mixture models (GMMs) in $\mathrm{Sim}(3)$ space. Furthermore, we design a joint 3DGS registration module that integrates the $\text{MW}_2$ distance, photometric consistency, and depth geometry. This enables a coarse-to-fine registration process while accurately estimating camera poses and aligning the scene. Experiments on the RE10K and ACID datasets demonstrate that RegGS effectively registers local Gaussians with high fidelity, achieving precise pose estimation and high-quality novel-view synthesis. Project page: https://3dagentworld.github.io/reggs/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07000v1">Enhancing non-Rigid 3D Model Deformations Using Mesh-based Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      We propose a novel framework that enhances non-rigid 3D model deformations by bridging mesh representations with 3D Gaussian splatting. While traditional Gaussian splatting delivers fast, real-time radiance-field rendering, its post-editing capabilities and support for large-scale, non-rigid deformations remain limited. Our method addresses these challenges by embedding Gaussian kernels directly onto explicit mesh surfaces. This allows the mesh's inherent topological and geometric priors to guide intuitive editing operations -- such as moving, scaling, and rotating individual 3D components -- and enables complex deformations like bending and stretching. This work paves the way for more flexible 3D content-creation workflows in applications spanning virtual reality, character animation, and interactive design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06927v3">CULTURE3D: A Large-Scale and Diverse Dataset of Cultural Landmarks and Terrains for Gaussian-Based Scene Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Current state-of-the-art 3D reconstruction models face limitations in building extra-large scale outdoor scenes, primarily due to the lack of sufficiently large-scale and detailed datasets. In this paper, we present a extra-large fine-grained dataset with 10 billion points composed of 41,006 drone-captured high-resolution aerial images, covering 20 diverse and culturally significant scenes from worldwide locations such as Cambridge Uni main buildings, the Pyramids, and the Forbidden City Palace. Compared to existing datasets, ours offers significantly larger scale and higher detail, uniquely suited for fine-grained 3D applications. Each scene contains an accurate spatial layout and comprehensive structural information, supporting detailed 3D reconstruction tasks. By reconstructing environments using these detailed images, our dataset supports multiple applications, including outputs in the widely adopted COLMAP format, establishing a novel benchmark for evaluating state-of-the-art large-scale Gaussian Splatting methods.The dataset's flexibility encourages innovations and supports model plug-ins, paving the way for future 3D breakthroughs. All datasets and code will be open-sourced for community use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04004v2">Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      This paper presents the first photo-realistic LiDAR-Inertial-Camera Gaussian Splatting SLAM system that simultaneously addresses visual quality, geometric accuracy, and real-time performance. The proposed method performs robust and accurate pose estimation within a continuous-time trajectory optimization framework, while incrementally reconstructing a 3D Gaussian map using camera and LiDAR data, all in real time. The resulting map enables high-quality, real-time novel view rendering of both RGB images and depth maps. To effectively address under-reconstruction in regions not covered by the LiDAR, we employ a lightweight zero-shot depth model that synergistically combines RGB appearance cues with sparse LiDAR measurements to generate dense depth maps. The depth completion enables reliable Gaussian initialization in LiDAR-blind areas, significantly improving system applicability for sparse LiDAR sensors. To enhance geometric accuracy, we use sparse but precise LiDAR depths to supervise Gaussian map optimization and accelerate it with carefully designed CUDA-accelerated strategies. Furthermore, we explore how the incrementally reconstructed Gaussian map can improve the robustness of odometry. By tightly incorporating photometric constraints from the Gaussian map into the continuous-time factor graph optimization, we demonstrate improved pose estimation under LiDAR degradation scenarios. We also showcase downstream applications via extending our elaborate system, including video frame interpolation and fast 3D mesh extraction. To support rigorous evaluation, we construct a dedicated LiDAR-Inertial-Camera dataset featuring ground-truth poses, depth maps, and extrapolated trajectories for assessing out-of-sequence novel view synthesis. Both the dataset and code will be made publicly available on project page https://xingxingzuo.github.io/gaussian_lic2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12510v2">PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Endoluminal endoscopic procedures are essential for diagnosing colorectal cancer and other severe conditions in the digestive tract, urogenital system, and airways. 3D reconstruction and novel-view synthesis from endoscopic images are promising tools for enhancing diagnosis. Moreover, integrating physiological deformations and interaction with the endoscope enables the development of simulation tools from real video data. However, constrained camera trajectories and view-dependent lighting create artifacts, leading to inaccurate or overfitted reconstructions. We present PR-ENDO, a novel 3D reconstruction framework leveraging the unique property of endoscopic imaging, where a single light source is closely aligned with the camera. Our method separates light effects from tissue properties. PR-ENDO enhances 3D Gaussian Splatting with a physically based relightable model. We boost the traditional light transport formulation with a specialized MLP capturing complex light-related effects while ensuring reduced artifacts and better generalization across novel views. PR-ENDO achieves superior reconstruction quality compared to baseline methods on both public and in-house datasets. Unlike existing approaches, PR-ENDO enables tissue modifications while preserving a physically accurate response to light, making it closer to real-world clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06684v1">Photometric Stereo using Gaussian Splatting and inverse rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 in French language. GRETSI 2025, Association GRETSI, Aug 2025, Strasbourg, France
    </div>
    <details class="paper-abstract">
      Recent state-of-the-art algorithms in photometric stereo rely on neural networks and operate either through prior learning or inverse rendering optimization. Here, we revisit the problem of calibrated photometric stereo by leveraging recent advances in 3D inverse rendering using the Gaussian Splatting formalism. This allows us to parameterize the 3D scene to be reconstructed and optimize it in a more interpretable manner. Our approach incorporates a simplified model for light representation and demonstrates the potential of the Gaussian Splatting rendering engine for the photometric stereo problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06671v1">FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 To appear at ACM MM 2025
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting has become a prominent technique for representing and rendering complex 3D scenes, due to its high fidelity and speed advantages. However, the growing demand for large-scale models calls for effective compression to reduce memory and computation costs, especially on mobile and edge devices with limited resources. Existing compression methods effectively reduce 3D Gaussian parameters but often require extensive retraining or fine-tuning, lacking flexibility under varying compression constraints. In this paper, we introduce FlexGaussian, a flexible and cost-effective method that combines mixed-precision quantization with attribute-discriminative pruning for training-free 3D Gaussian compression. FlexGaussian eliminates the need for retraining and adapts easily to diverse compression targets. Evaluation results show that FlexGaussian achieves up to 96.4% compression while maintaining high rendering quality (<1 dB drop in PSNR), and is deployable on mobile devices. FlexGaussian delivers high compression ratios within seconds, being 1.7-2.1x faster than state-of-the-art training-free methods and 10-100x faster than training-involved approaches. The code is being prepared and will be released soon at: https://github.com/Supercomputing-System-AI-Lab/FlexGaussian
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06647v1">ClipGS: Clippable Gaussian Splatting for Interactive Cinematic Visualization of Volumetric Medical Data</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Early accepted by MICCAI 2025. Project is available at: https://med-air.github.io/ClipGS
    </div>
    <details class="paper-abstract">
      The visualization of volumetric medical data is crucial for enhancing diagnostic accuracy and improving surgical planning and education. Cinematic rendering techniques significantly enrich this process by providing high-quality visualizations that convey intricate anatomical details, thereby facilitating better understanding and decision-making in medical contexts. However, the high computing cost and low rendering speed limit the requirement of interactive visualization in practical applications. In this paper, we introduce ClipGS, an innovative Gaussian splatting framework with the clipping plane supported, for interactive cinematic visualization of volumetric medical data. To address the challenges posed by dynamic interactions, we propose a learnable truncation scheme that automatically adjusts the visibility of Gaussian primitives in response to the clipping plane. Besides, we also design an adaptive adjustment model to dynamically adjust the deformation of Gaussians and refine the rendering performance. We validate our method on five volumetric medical data (including CT and anatomical slice data), and reach an average 36.635 PSNR rendering quality with 156 FPS and 16.1 MB model size, outperforming state-of-the-art methods in rendering quality and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18133v2">Gaussian Fluids: A Grid-Free Fluid Solver based on Gaussian Spatial Representation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      We present a grid-free fluid solver featuring a novel Gaussian representation. Drawing inspiration from the expressive capabilities of 3D Gaussian Splatting in multi-view image reconstruction, we model the continuous flow velocity as a weighted sum of multiple Gaussian functions. This representation is continuously differentiable, which enables us to derive spatial differentials directly and solve the time-dependent PDE via a custom first-order optimization tailored to fluid dynamics. Compared to traditional discretizations, which typically adopt Eulerian, Lagrangian, or hybrid perspectives, our approach is inherently memory-efficient and spatially adaptive, enabling it to preserve fine-scale structures and vortices with high fidelity. While these advantages are also sought by implicit neural representations, GSR offers enhanced robustness, accuracy, and generality across diverse fluid phenomena, with improved computational efficiency during temporal evolution. Though our first-order solver does not yet match the speed of fluid solvers using explicit representations, its continuous nature substantially reduces spatial discretization error and opens a new avenue for high-fidelity simulation. We evaluate the proposed solver across a broad range of 2D and 3D fluid phenomena, demonstrating its ability to preserve intricate vortex dynamics, accurately capture boundary-induced effects such as K\'arm\'an vortex streets, and remain robust across long time horizons - all without additional parameter tuning. Our results suggest that GSR offers a compelling direction for future research in fluid simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18394v4">Reconstructing Satellites in 3D from Amateur Telescope Images</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Monitoring space objects is crucial for space situational awareness, yet reconstructing 3D satellite models from ground-based telescope images is challenging due to atmospheric turbulence, long observation distances, limited viewpoints, and low signal-to-noise ratios. In this paper, we propose a novel computational imaging framework that overcomes these obstacles by integrating a hybrid image pre-processing pipeline with a joint pose estimation and 3D reconstruction module based on controlled Gaussian Splatting (GS) and Branch-and-Bound (BnB) search. We validate our approach on both synthetic satellite datasets and on-sky observations of China's Tiangong Space Station and the International Space Station, achieving robust 3D reconstructions of low-Earth orbit satellites from ground-based data. Quantitative evaluations using SSIM, PSNR, LPIPS, and Chamfer Distance demonstrate that our method outperforms state-of-the-art NeRF-based approaches, and ablation studies confirm the critical role of each component. Our framework enables high-fidelity 3D satellite monitoring from Earth, offering a cost-effective alternative for space situational awareness. Project page: https://ai4scientificimaging.org/ReconstructingSatellites
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15582v2">EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Acccpeted by ICCV2025
    </div>
    <details class="paper-abstract">
      Photorealistic reconstruction of street scenes is essential for developing real-world simulators in autonomous driving. While recent methods based on 3D/4D Gaussian Splatting (GS) have demonstrated promising results, they still encounter challenges in complex street scenes due to the unpredictable motion of dynamic objects. Current methods typically decompose street scenes into static and dynamic objects, learning the Gaussians in either a supervised manner (e.g., w/ 3D bounding-box) or a self-supervised manner (e.g., w/o 3D bounding-box). However, these approaches do not effectively model the motions of dynamic objects (e.g., the motion speed of pedestrians is clearly different from that of vehicles), resulting in suboptimal scene decomposition. To address this, we propose Explicit Motion Decomposition (EMD), which models the motions of dynamic objects by introducing learnable motion embeddings to the Gaussians, enhancing the decomposition in street scenes. The proposed plug-and-play EMD module compensates for the lack of motion modeling in self-supervised street Gaussian splatting methods. We also introduce tailored training strategies to extend EMD to supervised approaches. Comprehensive experiments demonstrate the effectiveness of our method, achieving state-of-the-art novel view synthesis performance in self-supervised settings. The code is available at: https://qingpowuwu.github.io/emd.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12981v2">GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted by ICCV2025
    </div>
    <details class="paper-abstract">
      Gaze estimation encounters generalization challenges when dealing with out-of-distribution data. To address this problem, recent methods use neural radiance fields (NeRF) to generate augmented data. However, existing methods based on NeRF are computationally expensive and lack facial details. 3D Gaussian Splatting (3DGS) has become the prevailing representation of neural fields. While 3DGS has been extensively examined in head avatars, it faces challenges with accurate gaze control and generalization across different subjects. In this work, we propose GazeGaussian, the first high-fidelity gaze redirection method that uses a two-stream 3DGS model to represent the face and eye regions separately. Leveraging the unstructured nature of 3DGS, we develop a novel representation of the eye for rigid eye rotation based on the target gaze direction. To enable synthesis generalization across various subjects, we integrate an expression-guided module to inject subject-specific information into the neural renderer. Comprehensive experiments show that GazeGaussian outperforms existing methods in rendering speed, gaze redirection accuracy, and facial synthesis across multiple datasets. The code is available at: https://ucwxb.github.io/GazeGaussian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14931v2">Hier-SLAM++: Neuro-Symbolic Semantic SLAM with a Hierarchically Categorical Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 18 pages. Under review
    </div>
    <details class="paper-abstract">
      We propose Hier-SLAM++, a comprehensive Neuro-Symbolic semantic 3D Gaussian Splatting SLAM method with both RGB-D and monocular input featuring an advanced hierarchical categorical representation, which enables accurate pose estimation as well as global 3D semantic mapping. The parameter usage in semantic SLAM systems increases significantly with the growing complexity of the environment, making scene understanding particularly challenging and costly. To address this problem, we introduce a novel hierarchical representation that encodes both semantic and geometric information in a compact form into 3D Gaussian Splatting, leveraging the capabilities of large language models (LLMs) as well as the 3D generative model. By utilizing the proposed hierarchical tree structure, semantic information is symbolically represented and learned in an end-to-end manner. We further introduce an advanced semantic loss designed to optimize hierarchical semantic information through both Intra-level and Inter-level optimizations. Additionally, we propose an improved SLAM system to support both RGB-D and monocular inputs using a feed-forward model. To the best of our knowledge, this is the first semantic monocular Gaussian Splatting SLAM system, significantly reducing sensor requirements for 3D semantic understanding and broadening the applicability of semantic Gaussian SLAM system. We conduct experiments on both synthetic and real-world datasets, demonstrating superior or on-par performance with state-of-the-art methods, while significantly reducing storage and training time requirements. Our project page is available at: https://hierslampp.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07136v1">LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Project Page: https://langsplat-v2.github.io
    </div>
    <details class="paper-abstract">
      In this paper, we introduce LangSplatV2, which achieves high-dimensional feature splatting at 476.2 FPS and 3D open-vocabulary text querying at 384.6 FPS for high-resolution images, providing a 42 $\times$ speedup and a 47 $\times$ boost over LangSplat respectively, along with improved query accuracy. LangSplat employs Gaussian Splatting to embed 2D CLIP language features into 3D, significantly enhancing speed and learning a precise 3D language field with SAM semantics. Such advancements in 3D language fields are crucial for applications that require language interaction within complex scenes. However, LangSplat does not yet achieve real-time inference performance (8.2 FPS), even with advanced A100 GPUs, severely limiting its broader application. In this paper, we first conduct a detailed time analysis of LangSplat, identifying the heavyweight decoder as the primary speed bottleneck. Our solution, LangSplatV2 assumes that each Gaussian acts as a sparse code within a global dictionary, leading to the learning of a 3D sparse coefficient field that entirely eliminates the need for a heavyweight decoder. By leveraging this sparsity, we further propose an efficient sparse coefficient splatting method with CUDA optimization, rendering high-dimensional feature maps at high quality while incurring only the time cost of splatting an ultra-low-dimensional feature. Our experimental results demonstrate that LangSplatV2 not only achieves better or competitive query accuracy but is also significantly faster. Codes and demos are available at our project page: https://langsplat-v2.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02803v2">HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Project page: https://gserifi.github.io/HyperGaussians, Code: https://github.com/gserifi/HyperGaussians
    </div>
    <details class="paper-abstract">
      We introduce HyperGaussians, a novel extension of 3D Gaussian Splatting for high-quality animatable face avatars. Creating such detailed face avatars from videos is a challenging problem and has numerous applications in augmented and virtual reality. While tremendous successes have been achieved for static faces, animatable avatars from monocular videos still fall in the uncanny valley. The de facto standard, 3D Gaussian Splatting (3DGS), represents a face through a collection of 3D Gaussian primitives. 3DGS excels at rendering static faces, but the state-of-the-art still struggles with nonlinear deformations, complex lighting effects, and fine details. While most related works focus on predicting better Gaussian parameters from expression codes, we rethink the 3D Gaussian representation itself and how to make it more expressive. Our insights lead to a novel extension of 3D Gaussians to high-dimensional multivariate Gaussians, dubbed 'HyperGaussians'. The higher dimensionality increases expressivity through conditioning on a learnable local embedding. However, splatting HyperGaussians is computationally expensive because it requires inverting a high-dimensional covariance matrix. We solve this by reparameterizing the covariance matrix, dubbed the 'inverse covariance trick'. This trick boosts the efficiency so that HyperGaussians can be seamlessly integrated into existing models. To demonstrate this, we plug in HyperGaussians into the state-of-the-art in fast monocular face avatars: FlashAvatar. Our evaluation on 19 subjects from 4 face datasets shows that HyperGaussians outperform 3DGS numerically and visually, particularly for high-frequency details like eyeglass frames, teeth, complex facial movements, and specular reflections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06109v1">LighthouseGS: Indoor Structure-aware 3D Gaussian Splatting for Panorama-Style Mobile Captures</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in 3D Gaussian Splatting (3DGS) have enabled real-time novel view synthesis (NVS) with impressive quality in indoor scenes. However, achieving high-fidelity rendering requires meticulously captured images covering the entire scene, limiting accessibility for general users. We aim to develop a practical 3DGS-based NVS framework using simple panorama-style motion with a handheld camera (e.g., mobile device). While convenient, this rotation-dominant motion and narrow baseline make accurate camera pose and 3D point estimation challenging, especially in textureless indoor scenes. To address these challenges, we propose LighthouseGS, a novel framework inspired by the lighthouse-like sweeping motion of panoramic views. LighthouseGS leverages rough geometric priors, such as mobile device camera poses and monocular depth estimation, and utilizes the planar structures often found in indoor environments. We present a new initialization method called plane scaffold assembly to generate consistent 3D points on these structures, followed by a stable pruning strategy to enhance geometry and optimization stability. Additionally, we introduce geometric and photometric corrections to resolve inconsistencies from motion drift and auto-exposure in mobile devices. Tested on collected real and synthetic indoor scenes, LighthouseGS delivers photorealistic rendering, surpassing state-of-the-art methods and demonstrating the potential for panoramic view synthesis and object placement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06103v1">Reflections Unlock: Geometry-Aware Reflection Disentanglement in 3D Gaussian Splatting for Photorealistic Scenes Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Accurately rendering scenes with reflective surfaces remains a significant challenge in novel view synthesis, as existing methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) often misinterpret reflections as physical geometry, resulting in degraded reconstructions. Previous methods rely on incomplete and non-generalizable geometric constraints, leading to misalignment between the positions of Gaussian splats and the actual scene geometry. When dealing with real-world scenes containing complex geometry, the accumulation of Gaussians further exacerbates surface artifacts and results in blurred reconstructions. To address these limitations, in this work, we propose Ref-Unlock, a novel geometry-aware reflection modeling framework based on 3D Gaussian Splatting, which explicitly disentangles transmitted and reflected components to better capture complex reflections and enhance geometric consistency in real-world scenes. Our approach employs a dual-branch representation with high-order spherical harmonics to capture high-frequency reflective details, alongside a reflection removal module providing pseudo reflection-free supervision to guide clean decomposition. Additionally, we incorporate pseudo-depth maps and a geometry-aware bilateral smoothness constraint to enhance 3D geometric consistency and stability in decomposition. Extensive experiments demonstrate that Ref-Unlock significantly outperforms classical GS-based reflection methods and achieves competitive results with NeRF-based models, while enabling flexible vision foundation models (VFMs) driven reflection editing. Our method thus offers an efficient and generalizable solution for realistic rendering of reflective scenes. Our code is available at https://ref-unlock.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06060v1">VisualSpeaker: Visually-Guided 3D Avatar Lip Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Realistic, high-fidelity 3D facial animations are crucial for expressive avatar systems in human-computer interaction and accessibility. Although prior methods show promising quality, their reliance on the mesh domain limits their ability to fully leverage the rapid visual innovations seen in 2D computer vision and graphics. We propose VisualSpeaker, a novel method that bridges this gap using photorealistic differentiable rendering, supervised by visual speech recognition, for improved 3D facial animation. Our contribution is a perceptual lip-reading loss, derived by passing photorealistic 3D Gaussian Splatting avatar renders through a pre-trained Visual Automatic Speech Recognition model during training. Evaluation on the MEAD dataset demonstrates that VisualSpeaker improves both the standard Lip Vertex Error metric by 56.1% and the perceptual quality of the generated animations, while retaining the controllability of mesh-driven animation. This perceptual focus naturally supports accurate mouthings, essential cues that disambiguate similar manual signs in sign language avatars.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22099v3">BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted at ICCV 2025, Project Page: https://github.com/fudan-zvg/BezierGS
    </div>
    <details class="paper-abstract">
      The realistic reconstruction of street scenes is critical for developing real-world simulators in autonomous driving. Most existing methods rely on object pose annotations, using these poses to reconstruct dynamic objects and move them during the rendering process. This dependence on high-precision object annotations limits large-scale and extensive scene reconstruction. To address this challenge, we propose B\'ezier curve Gaussian splatting (B\'ezierGS), which represents the motion trajectories of dynamic objects using learnable B\'ezier curves. This approach fully leverages the temporal information of dynamic objects and, through learnable curve modeling, automatically corrects pose errors. By introducing additional supervision on dynamic object rendering and inter-curve consistency constraints, we achieve reasonable and accurate separation and reconstruction of scene elements. Extensive experiments on the Waymo Open Dataset and the nuPlan benchmark demonstrate that B\'ezierGS outperforms state-of-the-art alternatives in both dynamic and static scene components reconstruction and novel view synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05859v1">D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 12 pages, 9 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Free-viewpoint video (FVV) enables immersive 3D experiences, but efficient compression of dynamic 3D representations remains a major challenge. Recent advances in 3D Gaussian Splatting (3DGS) and its dynamic extensions have enabled high-fidelity scene modeling. However, existing methods often couple scene reconstruction with optimization-dependent coding, which limits generalizability. This paper presents Feedforward Compression of Dynamic Gaussian Splatting (D-FCGS), a novel feedforward framework for compressing temporally correlated Gaussian point cloud sequences. Our approach introduces a Group-of-Frames (GoF) structure with I-P frame coding, where inter-frame motions are extracted via sparse control points. The resulting motion tensors are compressed in a feedforward manner using a dual prior-aware entropy model that combines hyperprior and spatial-temporal priors for accurate rate estimation. For reconstruction, we perform control-point-guided motion compensation and employ a refinement network to enhance view-consistent fidelity. Trained on multi-view video-derived Gaussian frames, D-FCGS generalizes across scenes without per-scene optimization. Experiments show that it matches the rate-distortion performance of optimization-based methods, achieving over 40 times compression in under 2 seconds while preserving visual quality across viewpoints. This work advances feedforward compression for dynamic 3DGS, paving the way for scalable FVV transmission and storage in immersive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05763v1">DreamArt: Generating Interactable Articulated Objects from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Generating articulated objects, such as laptops and microwaves, is a crucial yet challenging task with extensive applications in Embodied AI and AR/VR. Current image-to-3D methods primarily focus on surface geometry and texture, neglecting part decomposition and articulation modeling. Meanwhile, neural reconstruction approaches (e.g., NeRF or Gaussian Splatting) rely on dense multi-view or interaction data, limiting their scalability. In this paper, we introduce DreamArt, a novel framework for generating high-fidelity, interactable articulated assets from single-view images. DreamArt employs a three-stage pipeline: firstly, it reconstructs part-segmented and complete 3D object meshes through a combination of image-to-3D generation, mask-prompted 3D segmentation, and part amodal completion. Second, we fine-tune a video diffusion model to capture part-level articulation priors, leveraging movable part masks as prompt and amodal images to mitigate ambiguities caused by occlusion. Finally, DreamArt optimizes the articulation motion, represented by a dual quaternion, and conducts global texture refinement and repainting to ensure coherent, high-quality textures across all parts. Experimental results demonstrate that DreamArt effectively generates high-quality articulated objects, possessing accurate part shape, high appearance fidelity, and plausible articulation, thereby providing a scalable solution for articulated asset generation. Our project page is available at https://dream-art-0.github.io/DreamArt/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05661v1">3DGS_LSR:Large_Scale Relocation for Autonomous Driving Based on 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 13 pages,7 figures,4 tables
    </div>
    <details class="paper-abstract">
      In autonomous robotic systems, precise localization is a prerequisite for safe navigation. However, in complex urban environments, GNSS positioning often suffers from signal occlusion and multipath effects, leading to unreliable absolute positioning. Traditional mapping approaches are constrained by storage requirements and computational inefficiency, limiting their applicability to resource-constrained robotic platforms. To address these challenges, we propose 3DGS-LSR: a large-scale relocalization framework leveraging 3D Gaussian Splatting (3DGS), enabling centimeter-level positioning using only a single monocular RGB image on the client side. We combine multi-sensor data to construct high-accuracy 3DGS maps in large outdoor scenes, while the robot-side localization requires just a standard camera input. Using SuperPoint and SuperGlue for feature extraction and matching, our core innovation is an iterative optimization strategy that refines localization results through step-by-step rendering, making it suitable for real-time autonomous navigation. Experimental validation on the KITTI dataset demonstrates our 3DGS-LSR achieves average positioning accuracies of 0.026m, 0.029m, and 0.081m in town roads, boulevard roads, and traffic-dense highways respectively, significantly outperforming other representative methods while requiring only monocular RGB input. This approach provides autonomous robots with reliable localization capabilities even in challenging urban environments where GNSS fails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06269v1">A Probabilistic Approach to Uncertainty Quantification Leveraging 3D Geometry</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 ICCV 2025 Workshops (8 Pages, 6 Figures, 2 Tables)
    </div>
    <details class="paper-abstract">
      Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and awareness of fidelity surface geometric uncertainty are essential. Unlike radiance-based models such as NeRF or 3D Gaussian splatting, which lack explicit surface formulations, SDFs define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability via Hessian-based metrics, enabling computationally efficient, surface-aware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05256v1">SegmentDreamer: Towards High-fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 Accepted by ICCV 2025, project page: https://zjhjojo.github.io/
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-3D generation improve the visual quality of Score Distillation Sampling (SDS) and its variants by directly connecting Consistency Distillation (CD) to score distillation. However, due to the imbalance between self-consistency and cross-consistency, these CD-based methods inherently suffer from improper conditional guidance, leading to sub-optimal generation results. To address this issue, we present SegmentDreamer, a novel framework designed to fully unleash the potential of consistency models for high-fidelity text-to-3D generation. Specifically, we reformulate SDS through the proposed Segmented Consistency Trajectory Distillation (SCTD), effectively mitigating the imbalance issues by explicitly defining the relationship between self- and cross-consistency. Moreover, SCTD partitions the Probability Flow Ordinary Differential Equation (PF-ODE) trajectory into multiple sub-trajectories and ensures consistency within each segment, which can theoretically provide a significantly tighter upper bound on distillation error. Additionally, we propose a distillation pipeline for a more swift and stable generation. Extensive experiments demonstrate that our SegmentDreamer outperforms state-of-the-art methods in visual quality, enabling high-fidelity 3D asset creation through 3D Gaussian Splatting (3DGS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05040v3">GaussRender: Learning 3D Occupancy with Gaussian Rendering</a></div>
    <div class="paper-meta">
      📅 2025-07-07
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      Understanding the 3D geometry and semantics of driving scenes is critical for safe autonomous driving. Recent advances in 3D occupancy prediction have improved scene representation but often suffer from visual inconsistencies, leading to floating artifacts and poor surface localization. Existing voxel-wise losses (e.g., cross-entropy) fail to enforce visible geometric coherence. In this paper, we propose GaussRender, a module that improves 3D occupancy learning by enforcing projective consistency. Our key idea is to project both predicted and ground-truth 3D occupancy into 2D camera views, where we apply supervision. Our method penalizes 3D configurations that produce inconsistent 2D projections, thereby enforcing a more coherent 3D structure. To achieve this efficiently, we leverage differentiable rendering with Gaussian splatting. GaussRender seamlessly integrates with existing architectures while maintaining efficiency and requiring no inference-time modifications. Extensive evaluations on multiple benchmarks (SurroundOcc-nuScenes, Occ3D-nuScenes, SSCBench-KITTI360) demonstrate that GaussRender significantly improves geometric fidelity across various 3D occupancy models (TPVFormer, SurroundOcc, Symphonies), achieving state-of-the-art results, particularly on surface-sensitive metrics such as RayIoU. The code is open-sourced at https://github.com/valeoai/GaussRender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04961v1">InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting based 3D editing has demonstrated impressive performance in recent years. However, the multi-view editing often exhibits significant local inconsistency, especially in areas of non-rigid deformation, which lead to local artifacts, texture blurring, or semantic variations in edited 3D scenes. We also found that the existing editing methods, which rely entirely on text prompts make the editing process a "one-shot deal", making it difficult for users to control the editing degree flexibly. In response to these challenges, we present InterGSEdit, a novel framework for high-quality 3DGS editing via interactively selecting key views with users' preferences. We propose a CLIP-based Semantic Consistency Selection (CSCS) strategy to adaptively screen a group of semantically consistent reference views for each user-selected key view. Then, the cross-attention maps derived from the reference views are used in a weighted Gaussian Splatting unprojection to construct the 3D Geometry-Consistent Attention Prior ($GAP^{3D}$). We project $GAP^{3D}$ to obtain 3D-constrained attention, which are fused with 2D cross-attention via Attention Fusion Network (AFN). AFN employs an adaptive attention strategy that prioritizes 3D-constrained attention for geometric consistency during early inference, and gradually prioritizes 2D cross-attention maps in diffusion for fine-grained features during the later inference. Extensive experiments demonstrate that InterGSEdit achieves state-of-the-art performance, delivering consistent, high-fidelity 3DGS editing with improved user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05426v1">Mastering Regional 3DGS: Locating, Initializing, and Editing with Diverse 2D Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-07
    </div>
    <details class="paper-abstract">
      Many 3D scene editing tasks focus on modifying local regions rather than the entire scene, except for some global applications like style transfer, and in the context of 3D Gaussian Splatting (3DGS), where scenes are represented by a series of Gaussians, this structure allows for precise regional edits, offering enhanced control over specific areas of the scene; however, the challenge lies in the fact that 3D semantic parsing often underperforms compared to its 2D counterpart, making targeted manipulations within 3D spaces more difficult and limiting the fidelity of edits, which we address by leveraging 2D diffusion editing to accurately identify modification regions in each view, followed by inverse rendering for 3D localization, then refining the frontal view and initializing a coarse 3DGS with consistent views and approximate shapes derived from depth maps predicted by a 2D foundation model, thereby supporting an iterative, view-consistent editing process that gradually enhances structural details and textures to ensure coherence across perspectives. Experiments demonstrate that our method achieves state-of-the-art performance while delivering up to a $4\times$ speedup, providing a more efficient and effective approach to 3D scene local editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19420v3">RF-3DGS: Wireless Channel Modeling with Radio Radiance Field and 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-06
      | 💬 in submission to IEEE journals
    </div>
    <details class="paper-abstract">
      Precisely modeling radio propagation in complex environments has been a significant challenge, especially with the advent of 5G and beyond networks, where managing massive antenna arrays demands more detailed information. Traditional methods, such as empirical models and ray tracing, often fall short, either due to insufficient details or because of challenges for real-time applications. Inspired by the newly proposed 3D Gaussian Splatting method in the computer vision domain, which outperforms other methods in reconstructing optical radiance fields, we propose RF-3DGS, a novel approach that enables precise site-specific reconstruction of radio radiance fields from sparse samples. RF-3DGS can render radio spatial spectra at arbitrary positions within 2 ms following a brief 3-minute training period, effectively identifying dominant propagation paths. Furthermore, RF-3DGS can provide fine-grained Spatial Channel State Information (Spatial-CSI) of these paths, including the channel gain, the delay, the angle of arrival (AoA), and the angle of departure (AoD). Our experiments, calibrated through real-world measurements, demonstrate that RF-3DGS not only significantly improves reconstruction quality, training efficiency, and rendering speed compared to state-of-the-art methods, but also holds great potential for supporting wireless communication and advanced applications such as Integrated Sensing and Communication (ISAC). Code and dataset will be available at https://github.com/SunLab-UGA/RF-3DGS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00554v2">LOD-GS: Level-of-Detail-Sensitive 3D Gaussian Splatting for Detail Conserved Anti-Aliasing</a></div>
    <div class="paper-meta">
      📅 2025-07-06
    </div>
    <details class="paper-abstract">
      Despite the advancements in quality and efficiency achieved by 3D Gaussian Splatting (3DGS) in 3D scene rendering, aliasing artifacts remain a persistent challenge. Existing approaches primarily rely on low-pass filtering to mitigate aliasing. However, these methods are not sensitive to the sampling rate, often resulting in under-filtering and over-smoothing renderings. To address this limitation, we propose LOD-GS, a Level-of-Detail-sensitive filtering framework for Gaussian Splatting, which dynamically predicts the optimal filtering strength for each 3D Gaussian primitive. Specifically, we introduce a set of basis functions to each Gaussian, which take the sampling rate as input to model appearance variations, enabling sampling-rate-sensitive filtering. These basis function parameters are jointly optimized with the 3D Gaussian in an end-to-end manner. The sampling rate is influenced by both focal length and camera distance. However, existing methods and datasets rely solely on down-sampling to simulate focal length changes for anti-aliasing evaluation, overlooking the impact of camera distance. To enable a more comprehensive assessment, we introduce a new synthetic dataset featuring objects rendered at varying camera distances. Extensive experiments on both public datasets and our newly collected dataset demonstrate that our method achieves SOTA rendering quality while effectively eliminating aliasing. The code and dataset have been open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04147v1">A3FR: Agile 3D Gaussian Splatting with Incremental Gaze Tracked Foveated Rendering in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 ACM International Conference on Supercomputing 2025
    </div>
    <details class="paper-abstract">
      Virtual reality (VR) significantly transforms immersive digital interfaces, greatly enhancing education, professional practices, and entertainment by increasing user engagement and opening up new possibilities in various industries. Among its numerous applications, image rendering is crucial. Nevertheless, rendering methodologies like 3D Gaussian Splatting impose high computational demands, driven predominantly by user expectations for superior visual quality. This results in notable processing delays for real-time image rendering, which greatly affects the user experience. Additionally, VR devices such as head-mounted displays (HMDs) are intricately linked to human visual behavior, leveraging knowledge from perception and cognition to improve user experience. These insights have spurred the development of foveated rendering, a technique that dynamically adjusts rendering resolution based on the user's gaze direction. The resultant solution, known as gaze-tracked foveated rendering, significantly reduces the computational burden of the rendering process. Although gaze-tracked foveated rendering can reduce rendering costs, the computational overhead of the gaze tracking process itself can sometimes outweigh the rendering savings, leading to increased processing latency. To address this issue, we propose an efficient rendering framework called~\textit{A3FR}, designed to minimize the latency of gaze-tracked foveated rendering via the parallelization of gaze tracking and foveated rendering processes. For the rendering algorithm, we utilize 3D Gaussian Splatting, a state-of-the-art neural rendering technique. Evaluation results demonstrate that A3FR can reduce end-to-end rendering latency by up to $2\times$ while maintaining visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01110v2">A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Gaussian Splatting has emerged as a high-performance technique for novel view synthesis, enabling real-time rendering and high-quality reconstruction of small scenes. However, scaling to larger environments has so far relied on partitioning the scene into chunks -- a strategy that introduces artifacts at chunk boundaries, complicates training across varying scales, and is poorly suited to unstructured scenarios such as city-scale flyovers combined with street-level views. Moreover, rendering remains fundamentally limited by GPU memory, as all visible chunks must reside in VRAM simultaneously. We introduce A LoD of Gaussians, a framework for training and rendering ultra-large-scale Gaussian scenes on a single consumer-grade GPU -- without partitioning. Our method stores the full scene out-of-core (e.g., in CPU memory) and trains a Level-of-Detail (LoD) representation directly, dynamically streaming only the relevant Gaussians. A hybrid data structure combining Gaussian hierarchies with Sequential Point Trees enables efficient, view-dependent LoD selection, while a lightweight caching and view scheduling system exploits temporal coherence to support real-time streaming and rendering. Together, these innovations enable seamless multi-scale reconstruction and interactive visualization of complex scenes -- from broad aerial views to fine-grained ground-level details.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11868v2">MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Accurately analyzing the motion parts and their motion attributes in dynamic environments is crucial for advancing key areas such as embodied intelligence. Addressing the limitations of existing methods that rely on dense multi-view images or detailed part-level annotations, we propose an innovative framework that can analyze 3D mobility from monocular videos in a zero-shot manner. This framework can precisely parse motion parts and motion attributes only using a monocular video, completely eliminating the need for annotated training data. Specifically, our method first constructs the scene geometry and roughly analyzes the motion parts and their initial motion attributes combining depth estimation, optical flow analysis and point cloud registration method, then employs 2D Gaussian splatting for scene representation. Building on this, we introduce an end-to-end dynamic scene optimization algorithm specifically designed for articulated objects, refining the initial analysis results to ensure the system can handle 'rotation', 'translation', and even complex movements ('rotation+translation'), demonstrating high flexibility and versatility. To validate the robustness and wide applicability of our method, we created a comprehensive dataset comprising both simulated and real-world scenarios. Experimental results show that our framework can effectively analyze articulated object motions in an annotation-free manner, showcasing its significant potential in future embodied intelligence applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04004v1">Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      This paper proposes an innovative LiDAR-Inertial-Camera SLAM system with 3D Gaussian Splatting, which is the first to jointly consider visual quality, geometric accuracy, and real-time performance. It robustly and accurately estimates poses while building a photo-realistic 3D Gaussian map in real time that enables high-quality novel view RGB and depth rendering. To effectively address under-reconstruction in regions not covered by the LiDAR, we employ a lightweight zero-shot depth model that synergistically combines RGB appearance cues with sparse LiDAR measurements to generate dense depth maps. The depth completion enables reliable Gaussian initialization in LiDAR-blind areas, significantly improving system applicability for sparse LiDAR sensors. To enhance geometric accuracy, we use sparse but precise LiDAR depths to supervise Gaussian map optimization and accelerate it with carefully designed CUDA-accelerated strategies. Furthermore, we explore how the incrementally reconstructed Gaussian map can improve the robustness of odometry. By tightly incorporating photometric constraints from the Gaussian map into the continuous-time factor graph optimization, we demonstrate improved pose estimation under LiDAR degradation scenarios. We also showcase downstream applications via extending our elaborate system, including video frame interpolation and fast 3D mesh extraction. To support rigorous evaluation, we construct a dedicated LiDAR-Inertial-Camera dataset featuring ground-truth poses, depth maps, and extrapolated trajectories for assessing out-of-sequence novel view synthesis. Extensive experiments on both public and self-collected datasets demonstrate the superiority and versatility of our system across LiDAR sensors with varying sampling densities. Both the dataset and code will be made publicly available on project page https://xingxingzuo.github.io/gaussian_lic2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21420v2">EndoFlow-SLAM: Real-Time Endoscopic SLAM with Flow-Constrained Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 This paper has been accepted at MICCAI2025
    </div>
    <details class="paper-abstract">
      Efficient three-dimensional reconstruction and real-time visualization are critical in surgical scenarios such as endoscopy. In recent years, 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in efficient 3D reconstruction and rendering. Most 3DGS-based Simultaneous Localization and Mapping (SLAM) methods only rely on the appearance constraints for optimizing both 3DGS and camera poses. However, in endoscopic scenarios, the challenges include photometric inconsistencies caused by non-Lambertian surfaces and dynamic motion from breathing affects the performance of SLAM systems. To address these issues, we additionally introduce optical flow loss as a geometric constraint, which effectively constrains both the 3D structure of the scene and the camera motion. Furthermore, we propose a depth regularisation strategy to mitigate the problem of photometric inconsistencies and ensure the validity of 3DGS depth rendering in endoscopic scenes. In addition, to improve scene representation in the SLAM system, we improve the 3DGS refinement strategy by focusing on viewpoints corresponding to Keyframes with suboptimal rendering quality frames, achieving better rendering results. Extensive experiments on the C3VD static dataset and the StereoMIS dynamic dataset demonstrate that our method outperforms existing state-of-the-art methods in novel view synthesis and pose estimation, exhibiting high performance in both static and dynamic surgical scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03886v1">ArmGS: Composite Gaussian Appearance Refinement for Modeling Dynamic Urban Environments</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      This work focuses on modeling dynamic urban environments for autonomous driving simulation. Contemporary data-driven methods using neural radiance fields have achieved photorealistic driving scene modeling, but they suffer from low rendering efficacy. Recently, some approaches have explored 3D Gaussian splatting for modeling dynamic urban scenes, enabling high-fidelity reconstruction and real-time rendering. However, these approaches often neglect to model fine-grained variations between frames and camera viewpoints, leading to suboptimal results. In this work, we propose a new approach named ArmGS that exploits composite driving Gaussian splatting with multi-granularity appearance refinement for autonomous driving scene modeling. The core idea of our approach is devising a multi-level appearance modeling scheme to optimize a set of transformation parameters for composite Gaussian refinement from multiple granularities, ranging from local Gaussian level to global image level and dynamic actor level. This not only models global scene appearance variations between frames and camera viewpoints, but also models local fine-grained changes of background and objects. Extensive experiments on multiple challenging autonomous driving datasets, namely, Waymo, KITTI, NOTR and VKITTI2, demonstrate the superiority of our approach over the state-of-the-art methods.
    </details>
</div>
